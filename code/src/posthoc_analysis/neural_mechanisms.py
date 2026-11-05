"""Raw-data extraction for literature-guided neural mechanism exploration."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import signal, stats
from .config import PROJECT_ROOT, EXPECTED_SUBJECTS, get_subject_group
from .analysis import load_training_analysis_file
from .triggers import load_training_trigger_file


def load_neural_settings(path):
    """Require explicit, frozen preprocessing and acquisition assumptions."""
    settings=json.loads(Path(path).read_text())
    required=['native_to_uv','posterior_roi','frontal_roi','pre_spectral_window_s',
              'post_spectral_window_s','alpha_band_hz','theta_band_hz','roi_peak_to_peak_limit_uv',
              'erp_roi','erp_epoch_s','erp_filter_hz','erp_baseline_s','artifact_sensitivity_limits_uv']
    if any(k not in settings for k in required) or settings['native_to_uv']<=0:
        raise ValueError('Missing/invalid neural settings.')
    if 'FCZ' in settings['frontal_roi']:
        raise ValueError('Do not use the reported acquisition-reference channel in the frontal ROI.')
    if settings.get('artifact_policy_version') != 'filtered_roi_v2':
        raise ValueError('Use the user-requested filtered ROI screening settings, version filtered_roi_v2.')
    if settings['erp_filter_hz'] != [0.1, 20] or settings['erp_baseline_s'] != [-0.2, 0.0]:
        raise ValueError('Expected the documented Pd filter (0.1–20 Hz) and baseline (−0.2–0 s).')
    if settings['roi_peak_to_peak_limit_uv'] != 150 or settings['artifact_sensitivity_limits_uv'] != [100, 150, 200]:
        raise ValueError('Expected 150 µV primary and 100/200 µV sensitivity thresholds.')
    print('Loaded explicit exploratory neural settings; units follow user-endorsed prior plots.')
    return settings


def screen_filtered_roi_epochs(epochs, time, labels, settings):
    """Flag peak-to-peak excursions only in each measure's filtered EEG ROI.

    Input is trial x channel x time, already continuously bandpass filtered and
    baseline corrected. This is general amplitude QC, not eye-event detection.
    No samples or trials are removed. No linear detrending is performed.
    """
    epochs = np.asarray(epochs)
    time = np.asarray(time)
    if epochs.ndim != 3 or len(epochs) == 0 or epochs.shape[1:] != (len(labels), len(time)):
        raise ValueError('Expected trial x channel x time epochs with aligned labels and time.')
    if len(set(labels)) != len(labels) or not np.isfinite(epochs).all():
        raise ValueError('Duplicate channel labels or nonfinite EEG samples.')
    if len(time) < 2 or not np.allclose(np.diff(time), 1 / 512):
        raise ValueError('Screening requires a uniformly sampled 512 Hz time axis.')
    specs = {
        'pre_alpha': (settings['posterior_roi'], settings['pre_spectral_window_s']),
        'pre_theta': (settings['frontal_roi'], settings['pre_spectral_window_s']),
        'post_theta': (settings['frontal_roi'], settings['post_spectral_window_s']),
        'pd_epoch': (settings['erp_roi'], settings['erp_epoch_s']),
    }
    result = pd.DataFrame(index=np.arange(len(epochs)))
    for name, (channels, window) in specs.items():
        if not channels or not set(channels).issubset(labels):
            raise ValueError(f'{name}: missing/empty analysis electrodes {channels}.')
        if set(channels).intersection({'FP1', 'FP2', 'FPZ', 'EOG', 'sens7', 'sens8'}):
            raise ValueError('Ocular-proxy electrodes cannot determine ROI rejection.')
        lo, hi = window
        if not time[0] <= lo < hi <= time[-1] + 1 / 512 + 1e-12:
            raise ValueError(f'{name}: screening window {window} is outside the epochs.')
        mask = (time >= lo) & (time < hi)
        if mask.sum() < 2:
            raise ValueError(f'{name}: fewer than two screening samples.')
        ci = [labels.index(channel) for channel in channels]
        peak_to_peak = np.ptp(epochs[:, ci][:, :, mask], axis=-1).max(axis=1)
        result[f'{name}_ptp_uv'] = peak_to_peak
        for limit in settings['artifact_sensitivity_limits_uv']:
            result[f'{name}_pass_{limit}'] = peak_to_peak <= limit
    for limit in settings['artifact_sensitivity_limits_uv']:
        result[f'theta_pass_{limit}'] = result[f'pre_theta_pass_{limit}'] & result[f'post_theta_pass_{limit}']
    print(f'Filtered ROI amplitude QC: {len(epochs)} trials; '
          f'Pd PO7/PO8 {result.pd_epoch_pass_150.sum()}, '
          f'prestimulus alpha {result.pre_alpha_pass_150.sum()}, '
          f'theta {result.theta_pass_150.sum()} pass 150 µV peak-to-peak. No trials deleted.')
    return result


def extract_neural_run(gdf_path, settings):
    """Validate raw files and return aligned real trial features and posterior ERP."""
    import mne
    path=Path(gdf_path)
    analysis=load_training_analysis_file(path.with_suffix('.analysis.txt'))
    triggers=load_training_trigger_file(path.with_suffix('.triggers.txt'))
    raw=mne.io.read_raw_gdf(path,preload=True,verbose='ERROR')
    if raw.info['sfreq']!=512 or len(raw.ch_names)!=67 or len(set(raw.ch_names))!=67:
        raise ValueError(f'{path}: expected 512 Hz and 67 unique channels.')
    fs=512
    events=mne.find_events(raw,stim_channel='Status',shortest_event=1,verbose='ERROR')
    stimuli=events[np.isin(events[:,2],[8,32,44])]
    ts=triggers.loc[triggers.trigger.isin([8,32,44])].reset_index(drop=True)
    expected_codes=analysis.distractor_position.map({0:8,2:32,4:44}).to_numpy()
    if stimuli.shape!=(60,3) or not np.array_equal(stimuli[:,2],ts.trigger) or not np.array_equal(stimuli[:,2],expected_codes):
        raise ValueError('GDF/text/analysis stimulus condition and trial-order mismatch.')
    if not analysis.task.eq(analysis.distractor_position.ne(0).astype(int)).all():
        raise ValueError('Task labels disagree with distractor positions.')
    if not analysis.loc[analysis.task.eq(1),'target_position'].isin([1,3]).all():
        raise ValueError('Distractor-present target is not on the documented vertical midline.')
    if dict(zip(*np.unique(stimuli[:,2],return_counts=True)))!={8:30,32:15,44:15}:
        raise ValueError('Unbalanced stimulus counts, expected 30/15/15.')
    clock=stats.linregress(stimuli[:,0]/fs,ts.time)
    residual=(ts.time-(clock.intercept+clock.slope*stimuli[:,0]/fs))/clock.slope*1000
    if abs(clock.slope-1000)>1 or np.max(np.abs(residual))>10:
        raise ValueError(f'Clock calibration inconsistent with task milliseconds: slope={clock.slope}, residual={np.max(np.abs(residual))}')
    fixation=events[events[:,2]==4,0]
    previous_index=np.searchsorted(fixation,stimuli[:,0],side='left')-1
    if np.any(previous_index<0):raise ValueError('Missing preceding fixation.')
    fixation_s=(stimuli[:,0]-fixation[previous_index])/fs
    if np.any(fixation_s<.9):raise ValueError('Preparatory window not fully within fixation.')
    labels=[c for c in raw.ch_names if c not in ['M1','M2','EOG','sens7','sens8','Status']]
    if len(labels)!=61:raise ValueError('Expected 61 scalp channels.')
    scalp=raw.get_data(picks=labels)*settings['native_to_uv']
    if not np.isfinite(scalp).all():raise ValueError('Nonfinite continuous scalp data.')
    offsets=np.arange(-512,512,dtype=int)
    indices=stimuli[:,0,None]+offsets[None,:]
    if indices.min()<0 or indices.max()>=raw.n_times:raise ValueError('Incomplete raw epoch boundaries.')
    t=offsets/fs
    # Same continuous FIR filter and baseline helper as the existing Pd pipeline.
    # Extra prestimulus storage supports alpha; Pd retains its canonical −0.5–1 s epoch.
    from .eeg import baseline_correct_epochs
    filtered=mne.filter.filter_data(scalp,fs,*settings['erp_filter_hz'],method='fir',
                                   phase='zero',fir_design='firwin',verbose='ERROR')
    corrected=baseline_correct_epochs(filtered[:,indices].transpose(0,2,1),t,
                                     baseline_tmin=settings['erp_baseline_s'][0],
                                     baseline_tmax=settings['erp_baseline_s'][1])
    erp=corrected['baseline_corrected_epochs'].transpose(2,0,1)
    epochs=erp
    avg=epochs[:,[i for i,c in enumerate(labels) if c!='FCZ']].mean(axis=1,keepdims=True)
    row=analysis.copy()
    row['stimulus_code']=stimuli[:,2];row['stimulus_sample']=stimuli[:,0]
    row=pd.concat([row,screen_filtered_roi_epochs(erp,t,labels,settings)],axis=1)
    row['fixation_duration_s']=fixation_s
    # Authoritative task code writes milliseconds. Do not divide text timestamps by 512.
    row['text_rt_ms']=triggers.time.to_numpy().reshape(60,3)[:,2]-triggers.time.to_numpy().reshape(60,3)[:,1]
    row['gdf_rt_ms']=np.nan
    for i,onset in enumerate(stimuli[:,0]):
        end=stimuli[i+1,0] if i<59 else raw.n_times
        responses=events[(events[:,2]==64)&(events[:,0]>onset)&(events[:,0]<end),0]
        if len(responses)==1:row.loc[i,'gdf_rt_ms']=(responses[0]-onset)/fs*1000
        elif len(responses)>1:raise ValueError(f'Multiple response Status events for trial {i+1}.')
    available=row.gdf_rt_ms.notna()
    if np.any(np.abs(row.loc[available,'gdf_rt_ms']-row.loc[available,'text_rt_ms'])>20):
        raise ValueError('Text/GDF response intervals differ by more than 20 ms.')
    row['rt_ms']=row.gdf_rt_ms.where(available,row.text_rt_ms)
    row['rt_source']=np.where(available,'gdf_status_512_hz','text_milliseconds_fallback')
    spectra = {}
    for reference,arr in [('recorded',epochs),('average',epochs-avg)]:
        for window_name,window in [('pre',settings['pre_spectral_window_s']),('post',settings['post_spectral_window_s'])]:
            mask=(t>=window[0])&(t<window[1])
            if mask.sum()!=384:raise ValueError('Spectral windows must contain exactly 384 samples.')
            f,psd=signal.periodogram(arr[:,:,mask],fs=fs,window='hann',detrend=False,axis=-1,scaling='density')
            if not np.isfinite(psd).all() or (psd<0).any():raise ValueError('Invalid trial PSD.')
            for roi_name, channels in [('posterior', settings['posterior_roi']), ('frontal', settings['frontal_roi'])]:
                ci = [labels.index(c) for c in channels]
                spectra[f'{window_name}_{roi_name}_psd_{reference}'] = psd[:, ci].mean(axis=1).astype('float32')
            for band_name,band,channels in [('alpha',settings['alpha_band_hz'],settings['posterior_roi']),
                                            ('theta',settings['theta_band_hz'],settings['frontal_roi'])]:
                fi=(f>=band[0])&(f<=band[1]);ci=[labels.index(c) for c in channels]
                power=psd[:,ci][:,:,fi].sum(axis=-1).mean(axis=1)*(f[1]-f[0])
                if np.any(power<=0):raise ValueError('Nonpositive spectral ROI power.')
                row[f'{window_name}_{band_name}_log10_{reference}']=np.log10(power)
    baseline=(t>=-.2)&(t<0)
    if np.abs(erp[:,:,baseline].mean(axis=-1)).max()>1e-8:raise ValueError('ERP baseline residual.')
    po=erp[:,[labels.index('PO7'),labels.index('PO8')]]
    pd_sign=np.where(row.distractor_position==2,1,np.where(row.distractor_position==4,-1,np.nan))
    pdwave=(po[:,0]-po[:,1])*pd_sign[:,None]
    for name,window in [('pd',[.2,.5]),('late',[.5,.9]),('early',[.18,.3])]:
        mask=(t>=window[0])&(t<=window[1])
        row[f'{name}_lateral_mean']=pdwave[:,mask].mean(axis=-1)
        row[f'{name}_bilateral_mean']=po[:,:,mask].mean(axis=(1,2))
    row['theta_log10_post_pre']=row.post_theta_log10_recorded-row.pre_theta_log10_recorded
    row['theta_log10_post_pre_average']=row.post_theta_log10_average-row.pre_theta_log10_average
    print(f'{path.name}: 60 aligned trials, {row.pd_epoch_pass_150.sum()} pass Pd ROI 150 uV screen, '
          f'{available.sum()} Status responses; clock={clock.slope:.4f} ticks/s.')
    # Retain compact posterior single-trial ERPs and late whole-scalp means for follow-up controls.
    late_topo=erp[:,:,(t>=.5)&(t<=.9)].mean(axis=-1)
    # Response epochs use the SAME stimulus baseline. Missing/boundary events
    # remain explicit NaNs, never synthetic responses or extrapolated EEG.
    response_offsets = np.arange(-205, 206)
    response_po = np.full((60, 2, len(response_offsets)), np.nan, dtype='float32')
    po_indices = [labels.index('PO7'), labels.index('PO8')]
    stimulus_baseline = filtered[po_indices][:, indices][:, :, baseline].mean(axis=-1)
    for i in np.flatnonzero(available):
        response_sample = stimuli[i, 0] + int(round(row.loc[i, 'gdf_rt_ms'] * fs / 1000))
        ri = response_sample + response_offsets
        if ri.min() >= 0 and ri.max() < raw.n_times:
            response_po[i] = filtered[po_indices][:, ri] - stimulus_baseline[:, i, None]
    row['response_epoch_available'] = np.isfinite(response_po).all(axis=(1, 2))
    row['response_roi_ptp_uv'] = np.where(row.response_epoch_available,
                                         np.ptp(response_po, axis=-1).max(axis=1), np.nan)
    print(f'Response epochs: {row.response_epoch_available.sum()}/60 complete; stimulus baseline retained.')
    return row,dict(posterior_erp_uv=po.astype('float32'),time=t,late_topography_uv=late_topo.astype('float32'),
                   pd_topography_uv=erp[:, :, (t>=.2)&(t<=.5)].mean(axis=-1).astype('float32'),
                   response_posterior_uv=response_po,response_time=response_offsets/fs,
                   spectral_frequencies=f,**spectra,
                   channel_labels=np.array(labels)),dict(clock_hz=clock.slope,clock_residual_ms=float(np.abs(residual).max()),
                                                         n_status_responses=int(available.sum()),n_pd_roi_pass_150=int(row.pd_epoch_pass_150.sum()))


def extract_neural_cohort(settings_path, output_dir, project_root=PROJECT_ROOT, subjects=None):
    """Checkpoint validated per-run outputs; never silently accept failed runs."""
    from .eeg import _get_training_run_gdf_files_for_session
    settings=load_neural_settings(settings_path)
    subjects=EXPECTED_SUBJECTS if subjects is None else list(subjects)
    if not subjects or not set(subjects).issubset(EXPECTED_SUBJECTS):raise ValueError('Unknown/empty cohort.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(Path(settings_path).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    frames=[];logs=[]
    for subject in subjects:
        for session in (1,5):
            files=_get_training_run_gdf_files_for_session(subject,session,project_root=project_root,allow_incomplete=True)
            expected=7 if (subject,session)==('e30',1) else {1:8,5:4}[session]
            if len(files)!=expected:raise ValueError('Undocumented run count.')
            for run,path in enumerate(files,1):
                stem=f'{subject}_s{session}_r{run:02d}'
                metadata=dict(subject_id=subject,group=get_subject_group(subject),session_id=session,run_id=run,
                              source=str(path),source_size=path.stat().st_size,source_mtime_ns=path.stat().st_mtime_ns,
                              analysis_sha=hashlib.sha256(path.with_suffix('.analysis.txt').read_bytes()).hexdigest(),
                              triggers_sha=hashlib.sha256(path.with_suffix('.triggers.txt').read_bytes()).hexdigest(),signature=signature)
                checkpoint=out/f'{stem}.json'
                try:
                    if checkpoint.is_file() and (out/f'{stem}.csv').is_file() and (out/f'{stem}.npz').is_file():
                        saved=json.loads(checkpoint.read_text())
                        if all(saved.get(k)==v for k,v in metadata.items()):
                            frame=pd.read_csv(out/f'{stem}.csv')
                            if len(frame)!=60:raise ValueError('Corrupt cached trial count.')
                            frames.append(frame);logs.append(saved);continue
                    frame,arrays,qc=extract_neural_run(path,settings)
                    for k in ['subject_id','group','session_id','run_id']:frame[k]=metadata[k]
                    frame.to_csv(out/f'{stem}.csv',index=False)
                    np.savez_compressed(out/f'{stem}.npz',**arrays)
                    metadata.update(qc);metadata['status']='pass'
                    checkpoint.write_text(json.dumps(metadata,indent=2))
                    frames.append(frame);logs.append(metadata)
                except Exception as exc:
                    metadata.update(status='failed',error=str(exc));logs.append(metadata)
                    print(f'FAILED {stem}: {exc}')
                pd.DataFrame(logs).to_csv(out.parent/'neural_run_log.csv',index=False)
    result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(out.parent/'neural_trial_features.csv',index=False)
    log=pd.DataFrame(logs);log.to_csv(out.parent/'neural_run_log.csv',index=False)
    print(f'Extraction finished: {len(result)} trial rows, {(log.status=="pass").sum()} valid runs, {(log.status=="failed").sum()} failures.')
    return result,log

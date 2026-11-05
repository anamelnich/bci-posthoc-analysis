"""Signal-level surface Laplacian before single-trial theta power estimation."""


from pathlib import Path


from functools import lru_cache


from concurrent.futures import ThreadPoolExecutor


import hashlib


import inspect


import json


import time


import numpy as np


import pandas as pd


def make_laplacian_theta_settings(labels, base_settings):
    """Define the fixed template transform without requiring a historical cache."""
    import mne
    settings=dict(base_settings)
    if len(labels)!=61 or len(set(labels))!=61 or not settings.get('rt_cleaned'):
        raise ValueError('Expected 61 scalp labels and explicit retained-trial settings.')
    info=mne.create_info(labels,512,'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False)
    radius,origin,_=mne.bem.fit_sphere_to_headshape(info,dig_kinds='eeg',verbose='ERROR')
    sphere=np.r_[origin,radius]
    if not np.isfinite(sphere).all() or not .06<radius<.14:
        raise ValueError('Implausible template sphere fit.')
    lap=dict(sphere_m=sphere.tolist(),lambda2=1e-5,stiffness=4,n_legendre_terms=50,
        montage='standard_1020',geometry='sphere fitted to the 61 template EEG positions in head coordinates; identical for every run',
        channels=labels,reference_policy='61 recorded scalp channels; no zero-valued CPz channel added',
        order='continuous temporal filtering -> surface Laplacian -> single-trial epochs -> Morlet power -> pooled power baseline -> dB',
        mne_version=mne.__version__)
    settings.update(surface_laplacian=lap,reference='surface Laplacian of CPz-referenced voltage',
        cache_computation='Direct retained-trial power from Laplacian-transformed EEG; existing voltage power cannot be transformed into Laplacian power',
        method_scope='Reference-independent surface-Laplacian total theta power; template geometry; no artifact correction')
    matrix,transform_qc=_laplacian_matrix(tuple(labels),tuple(sphere),1e-5,4,50)
    print(f'Laplacian geometry validated: 61 channels; fitted template sphere {sphere.tolist()}.')
    return settings,matrix,transform_qc


@lru_cache(maxsize=4)
def _laplacian_matrix(labels, sphere, lambda2, stiffness, terms):
    """Construct MNE's transform and check an independent augmented-system solution."""
    import mne
    from numpy.polynomial.legendre import legval
    if len(labels)!=61 or len(set(labels))!=61 or 'CPZ' in labels or 'FCZ' not in labels:
        raise ValueError('Laplacian requires the same 61 recorded scalp channels.')
    info=mne.create_info(list(labels),512,'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False)
    basis=mne.io.RawArray(np.eye(len(labels)),info,verbose='ERROR')
    matrix=mne.preprocessing.compute_current_source_density(basis,sphere=sphere,
        lambda2=lambda2,stiffness=stiffness,n_legendre_terms=terms,verbose='ERROR').get_data()
    xyz=np.array([ch['loc'][:3] for ch in info['chs']])-np.array(sphere[:3])
    xyz/=np.linalg.norm(xyz,axis=1)[:,None];cosine=np.clip(xyz@xyz.T,-1,1)
    n=np.arange(1,terms+1,dtype=float)
    g=legval(cosine,np.r_[0,(2*n+1)/(4*np.pi*(n*(n+1))**stiffness)])
    h=legval(cosine,np.r_[0,(2*n+1)/(4*np.pi*(n*(n+1))**(stiffness-1))])
    count=len(labels);system=np.block([[g+lambda2*np.eye(count),np.ones((count,1))],[np.ones((1,count)),np.zeros((1,1))]])
    coef=np.linalg.solve(system,np.r_[np.eye(count),np.zeros((1,count))])
    independent=h@coef[:-1]/sphere[3]**2
    error=float(np.max(abs(matrix-independent))/np.max(abs(matrix)))
    common=float(np.max(abs(matrix@np.ones(count)))/np.max(abs(matrix)))
    if not np.isfinite(matrix).all() or max(error,common)>1e-9:
        raise ValueError(f'Laplacian validation failed: independent={error}, common-mode={common}.')
    qc=dict(relative_independent_matrix_error=error,relative_common_mode_residual=common,
        shape=list(matrix.shape),sphere_m=list(sphere),lambda2=lambda2,stiffness=stiffness,n_legendre_terms=terms,
        interpretation='Annihilates common reference; regularized spherical second spatial derivative, not image interpolation')
    matrix.setflags(write=False)
    print(f'Laplacian operator validated against independent equations; relative error {error:.3g}, common-mode residual {common:.3g}.',flush=True)
    return matrix,qc


def apply_theta_laplacian(filtered, labels, settings):
    """Spatially transform filtered voltage samples before nonlinear power estimation."""
    if labels!=settings['channels'] or filtered.ndim!=2 or filtered.shape[0]!=61 or not np.isfinite(filtered).all():
        raise ValueError('Invalid full-montage input or changed channel order.')
    matrix,qc=_laplacian_matrix(tuple(labels),tuple(settings['sphere_m']),settings['lambda2'],settings['stiffness'],settings['n_legendre_terms'])
    result=matrix@filtered
    voltage_rms=np.sqrt(np.mean(filtered**2,axis=1));laplacian_rms=np.sqrt(np.mean(result**2,axis=1))
    if not np.isfinite(result).all() or min(voltage_rms.min(),laplacian_rms.min())<=0:
        raise ValueError('Nonfinite or flat voltage/Laplacian channel.')
    audit=dict(laplacian_common_mode_residual=qc['relative_common_mode_residual'],
        voltage_rms=voltage_rms,laplacian_rms=laplacian_rms,
        maximum_voltage_rms_to_channel_median=float(max(voltage_rms/np.median(voltage_rms))),
        maximum_laplacian_rms_to_channel_median=float(max(laplacian_rms/np.median(laplacian_rms))))
    print(f'Applied signal-level Laplacian to {filtered.shape[1]:,} samples; all 61 channels finite.',flush=True)
    return result,audit


def _cache_laplacian_run(record, cell, settings, out):
    """Validate sources, process retained trials, and checkpoint sufficient statistics."""
    from .theta_topography import extract_theta_run
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source=Path(record['source'])
    key=f"{record['subject_id']}_s{record['session_id']}_r{record['run_id']:02d}"
    cell=cell.sort_values('trial');keep=cell.retained.to_numpy()
    if cell.trial.tolist()!=list(range(1,61)) or keep.dtype!=bool:
        raise ValueError(f'{key}: incorrect ordered 60-trial ledger.')
    stat=source.stat()
    if stat.st_size!=record['source_size'] or stat.st_mtime_ns!=record['source_mtime_ns']:
        raise ValueError(f'{key}: source GDF changed.')
    for suffix,field in [('.analysis.txt','analysis_sha'),('.triggers.txt','triggers_sha')]:
        if hashlib.sha256(source.with_suffix(suffix).read_bytes()).hexdigest()!=record[field]:
            raise ValueError(f'{key}: paired task file changed.')
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()+keep.tobytes()+cell.gdf_rt_ms.to_numpy().tobytes()
        +inspect.getsource(extract_theta_run).encode()+inspect.getsource(apply_theta_laplacian).encode()
        +inspect.getsource(_laplacian_matrix).encode()).hexdigest()
    npzpath=out/f'{key}.npz';jsonpath=out/f'{key}.json'
    if npzpath.exists() and jsonpath.exists():
        old=json.loads(jsonpath.read_text())
        source_matches=all(old.get(k)==record[k] for k in
            ['source','source_size','source_mtime_ns','analysis_sha','triggers_sha'])
        if source_matches and old.get('laplacian_signature')==signature and hashlib.sha256(npzpath.read_bytes()).hexdigest()==old['npz_sha']:
            if old['npz_path']!=str(npzpath.resolve()):
                old['npz_path']=str(npzpath.resolve())
                jsonpath.write_text(json.dumps(old,indent=2))
            print(f'{key}: Laplacian checkpoint validated/reused.',flush=True)
            return old
    start=time.monotonic()
    arrays,qc=extract_theta_run(source,settings,trial_mask=keep,expected_rt_ms=cell.gdf_rt_ms.to_numpy())
    expected=np.array([np.sum(keep&(cell.task.to_numpy()==c)) for c in (0,1)])
    if not np.array_equal(arrays['condition_counts'],expected):raise ValueError(f'{key}: retained counts differ.')
    arrays['retained_trial_numbers']=cell.loc[keep,'trial'].to_numpy(dtype=int)
    np.savez_compressed(npzpath,**arrays)
    meta={k:record[k] for k in ['subject_id','group','session_id','run_id','source','source_size','source_mtime_ns','analysis_sha','triggers_sha']}
    meta.update(qc,n_retained=int(keep.sum()),n_correct=int(keep.sum()),n_other=0,
        laplacian_signature=signature,npz_path=str(npzpath.resolve()),npz_sha=hashlib.sha256(npzpath.read_bytes()).hexdigest(),
        elapsed_s=time.monotonic()-start)
    jsonpath.write_text(json.dumps(meta,indent=2))
    print(f'{key}: CSD theta checkpoint complete ({meta["elapsed_s"]:.1f} s); {keep.sum()} retained trials.',flush=True)
    return meta


def build_laplacian_theta_cache(inventory_csv, ledger, settings, output_dir, n_jobs=2):
    """Recompute full-cohort Laplacian power from raw EEG with resumable run caches."""
    from .theta_topography import validate_theta_inventory
    if n_jobs not in (1,2):raise ValueError('Use one or two workers to bound memory.')
    inventory=validate_theta_inventory(inventory_csv)
    if len(ledger)!=22980 or ledger.retained.sum()!=21940 or not settings.get('surface_laplacian'):
        raise ValueError('Invalid fixed mask or missing Laplacian settings.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    grouped={k:v for k,v in ledger.groupby(['subject_id','session_id','run_id'])}
    def work(record):
        key=(record['subject_id'],record['session_id'],record['run_id'])
        return _cache_laplacian_run(record,grouped[key],settings,out/'runs')
    logs=[]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for result in pool.map(work,inventory.to_dict('records')):
            logs.append(result)
            pd.DataFrame(logs).to_csv(out/'run_log_progress.csv',index=False)
    log=pd.DataFrame(logs)
    if len(log)!=383 or log.n_retained.sum()!=21940 or log.n_excluded.sum()!=1040:
        raise ValueError('Laplacian run/trial totals do not reconcile.')
    log.to_csv(out/'run_log.csv',index=False)
    print('Laplacian theta complete: 383 runs, 32 participants, 21,940 identical retained trials.',flush=True)
    return log


def verify_laplacian_theta_pipeline(cache_dir):
    """Use two real retained trials to verify spatial and wavelet ordering independently."""
    import mne
    from scipy.signal import fftconvolve
    root=Path(cache_dir);settings=json.loads((root/'settings.json').read_text());lap=settings['surface_laplacian']
    log=pd.read_csv(root/'run_log.csv');ledger=pd.read_csv(root/'trial_mask.csv');run=log.iloc[0]
    cell=ledger[(ledger.subject_id==run.subject_id)&(ledger.session_id==run.session_id)&(ledger.run_id==run.run_id)].sort_values('trial')
    keep=np.flatnonzero(cell.retained.to_numpy())[:2]
    if len(keep)!=2 or len(log)!=383:raise ValueError('Verify after full-cohort completion with two retained trials.')
    raw=mne.io.read_raw_gdf(run.source,preload=True,verbose='ERROR');labels=lap['channels'];fs=settings['sfreq']
    if raw.info['sfreq']!=fs:raise ValueError('Validation sampling rate mismatch.')
    voltage=mne.filter.filter_data(raw.get_data(picks=labels),fs,*settings['filter_hz'],verbose='ERROR')
    events=mne.find_events(raw,stim_channel='Status',shortest_event=1,verbose='ERROR')
    onset=events[np.isin(events[:,2],[8,32,44]),0]-raw.first_samp
    offsets=np.arange(-fs,round(1.5*fs)+1);indices=onset[keep,None]+offsets
    epochs=voltage[:,indices].transpose(1,0,2)
    info=mne.create_info(labels,fs,'eeg');info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False)
    public=mne.EpochsArray(epochs,info,tmin=-1,baseline=None,verbose='ERROR')
    public=mne.preprocessing.compute_current_source_density(public,sphere=lap['sphere_m'],lambda2=lap['lambda2'],
        stiffness=lap['stiffness'],n_legendre_terms=lap['n_legendre_terms'],verbose='ERROR').get_data(copy=True)
    matrix=np.load(root/'laplacian_matrix.npy',allow_pickle=False)
    manual=np.einsum('ij,tjk->tik',matrix,epochs)
    spatial_error=float(np.linalg.norm(public-manual)/np.linalg.norm(public))
    freqs=np.array(settings['frequencies_hz']);cycles=np.array(settings['n_cycles'])
    waves=mne.time_frequency.morlet(fs,freqs,n_cycles=cycles,zero_mean=True)
    morlet=mne.time_frequency.tfr_array_morlet(manual,fs,freqs,n_cycles=cycles,zero_mean=True,use_fft=True,output='complex',verbose='ERROR')
    direct=np.stack([fftconvolve(manual,w[None,None,:],mode='same',axes=-1) for w in waves],axis=2)
    convolution_error=float(np.linalg.norm(morlet-direct)/np.linalg.norm(direct))
    voltage_analytic=mne.time_frequency.tfr_array_morlet(epochs,fs,freqs,n_cycles=cycles,zero_mean=True,use_fft=True,output='complex',verbose='ERROR')
    spatial_after_complex=np.einsum('ij,tjfk->tifk',matrix,voltage_analytic)
    commute_error=float(np.linalg.norm(morlet-spatial_after_complex)/np.linalg.norm(morlet))
    if max(spatial_error,convolution_error,commute_error)>1e-9:
        raise ValueError('Independent real-trial Laplacian/wavelet verification failed.')
    qc=dict(run=str(run.source),trial_numbers=(keep+1).tolist(),n_channels=61,n_frequencies=9,
        relative_public_epochs_error=spatial_error,relative_direct_convolution_error=convolution_error,
        relative_complex_wavelet_commutation_error=commute_error,
        ordering='Laplacian and linear complex wavelets commute; transform must precede squared magnitude, trial-power averaging and dB')
    (root/'pipeline_validation.json').write_text(json.dumps(qc,indent=2))
    print(f'Real-trial pipeline validation passed: maximum relative error {max(spatial_error,convolution_error,commute_error):.3g}.')
    return qc


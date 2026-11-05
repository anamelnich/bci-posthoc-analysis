"""Perrin spherical-spline display of scalp summaries (not a Laplacian)."""


import numpy as np


from numpy.polynomial.legendre import legval


def prepare_spherical_topography(info, radius=.095):
    """Project montage directions to a common sphere; preserve electrode data.

    Use the existing MNE head-coordinate origin (0, 0, 0), a 95-mm radius,
    and the same azimuthal-equidistant projection as MNE topomaps. Normalizing
    all radii makes the inverse display projection unambiguous. No EEG data
    are rereferenced or transformed.
    """
    if not np.isfinite(radius) or radius<=0 or len(info['chs'])<4:
        raise ValueError('A positive radius and at least four electrodes are required.')
    xyz=np.array([ch['loc'][:3] for ch in info['chs']])
    lengths=np.linalg.norm(xyz,axis=1)
    if not np.isfinite(xyz).all() or np.any(lengths<=0):
        raise ValueError('Spherical interpolation requires finite nonzero 3D electrode locations.')
    directions=xyz/lengths[:,None]
    if len(np.unique(np.round(directions,12),axis=0))!=len(xyz):
        raise ValueError('Duplicate spherical electrode directions.')
    plotted=info.copy()
    for ch,point in zip(plotted['chs'],directions*radius):ch['loc'][:3]=point
    print(f'Spherical display: {len(xyz)} electrodes projected radially to {radius*1000:g} mm; origin (0,0,0).')
    return plotted,directions


def spherical_spline_grid(directions, extent, shape, radius=.095, regularization=0.):
    """Return optionally regularized spherical-spline weights and validation.

    Perrin et al. (1989), doi:10.1016/0013-4694(89)90180-6:
    g(cos angle) = sum_n (2n+1) P_n / [4 pi (n(n+1))**4].
    Use 50 Legendre terms; positive diagonal regularization smooths the fit,
    while zero preserves historical exact interpolation. The system includes a free constant
    and sum(coefficients)=0. This is interpolation of dB summaries, not CSD.
    """
    from mne.channels.interpolation import _make_interpolation_matrix
    if not np.isscalar(regularization) or not np.isfinite(regularization) or regularization<0:
        raise ValueError('Regularization must be a finite nonnegative scalar.')
    directions=np.asarray(directions,float)
    if (directions.ndim!=2 or directions.shape[1]!=3 or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12)
            or len(shape)!=2 or min(shape)<2 or len(extent)!=4 or not np.isfinite(extent).all()
            or radius<=0):
        raise ValueError('Invalid unit directions, image grid, extent or sphere radius.')
    n=np.arange(1,51,dtype=float)
    factors=np.r_[0.,(2*n+1)/(4*np.pi*(n*(n+1))**4)]
    kernel=legval(np.clip(directions@directions.T,-1,1),factors)
    count=len(directions)
    system=np.block([[kernel+regularization*np.eye(count),np.ones((count,1))],[np.ones((1,count)),np.zeros((1,1))]])
    coefficients=np.linalg.solve(system,np.r_[np.eye(count),np.zeros((1,count))])
    node_weights=np.c_[kernel,np.ones(count)]@coefficients
    node_error=float(np.max(abs(node_weights-np.eye(count))))
    equation_error=float(np.max(abs(node_weights+regularization*coefficients[:-1]-np.eye(count))))
    xmin,xmax,ymin,ymax=extent
    xx,yy=np.meshgrid(np.linspace(xmin,xmax,shape[1]),np.linspace(ymin,ymax,shape[0]))
    azimuth=np.arctan2(yy,xx).ravel()
    polar=(np.hypot(xx,yy)*np.pi/(2*radius)).ravel()
    target=np.column_stack([np.sin(polar)*np.cos(azimuth),np.sin(polar)*np.sin(azimuth),np.cos(polar)])
    weights=np.c_[legval(np.clip(target@directions.T,-1,1),factors),np.ones(len(target))]@coefficients
    constant_error=float(np.max(abs(weights.sum(axis=1)-1)))
    # Independent implementation check at the same regularization strength.
    sample=np.linspace(0,len(target)-1,17,dtype=int)
    mne_error=float(np.max(abs(weights[sample]-_make_interpolation_matrix(directions,target[sample],alpha=regularization))))
    if not np.isfinite(weights).all() or max(equation_error,constant_error,mne_error)>1e-6:
        raise ValueError(f'Spherical spline validation failed: node={node_error}, constant={constant_error}, MNE={mne_error}.')
    qc=dict(method='Perrin spherical spline',stiffness=4,n_legendre_terms=50,regularization=float(regularization),
            sphere_origin_m=[0.,0.,0.],sphere_radius_m=radius,projection='azimuthal equidistant; montage radii normalized',
            maximum_node_weight_error=node_error,maximum_constant_error=constant_error,
            maximum_regularized_equation_error=equation_error,
            maximum_mne_weight_difference=mne_error,system_condition_number=float(np.linalg.cond(system)),
            source_doi='10.1016/0013-4694(89)90180-6',
            scope='Display interpolation of electrode dB summaries; no Laplacian, rereferencing or bad-channel replacement')
    print(f'Spherical spline checked: lambda={regularization:g}, equation error {equation_error:.3g}, constant error {constant_error:.3g}, MNE difference {mne_error:.3g}.')
    return weights,node_weights,qc


def inspect_spherical_theta_midpoint(table, output_dir, regularization=1e-5):
    """Audit post-training mental-rehearsal AF3--AF4 midpoint before/after smoothing."""
    from pathlib import Path
    import pandas as pd
    import mne
    from mne.channels.layout import _find_topomap_coords
    from mne.channels.interpolation import _make_interpolation_matrix
    required={'group','session_id','condition','channel','subject_id','theta_db'}
    if not required.issubset(table) or not np.isfinite(regularization) or regularization<=0:
        raise ValueError('Expected validated theta table and positive regularization.')
    sub=table[(table.group=='control')&(table.session_id==5)]
    labels=sorted(sub.channel.unique())
    if len(labels)!=61 or not {'AF3','AF4'}.issubset(labels) or not np.isfinite(sub.theta_db).all():
        raise ValueError('Expected all 61 finite scalp channels including AF3/AF4.')
    info=mne.create_info(labels,512,'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False)
    info,directions=prepare_spherical_topography(info)
    positions=_find_topomap_coords(info,np.arange(len(labels)),sphere=(0.,0.,0.,.095))
    midpoint=positions[[labels.index('AF3'),labels.index('AF4')]].mean(axis=0)
    polar=np.linalg.norm(midpoint)*np.pi/(2*.095);azimuth=np.arctan2(midpoint[1],midpoint[0])
    target=np.array([[np.sin(polar)*np.cos(azimuth),np.sin(polar)*np.sin(azimuth),np.cos(polar)]])
    weights=[_make_interpolation_matrix(directions,target,alpha=alpha)[0] for alpha in (0,regularization)]
    rows=[]
    for condition in ['distractor','no_distractor','both']:
        cell=sub[sub.condition==condition]
        if len(cell)!=16*61 or cell.duplicated(['subject_id','channel']).any() or not cell.groupby('channel').subject_id.nunique().eq(16).all():
            raise ValueError('Incomplete midpoint diagnostic participant/channel cell.')
        values=cell.groupby('channel').theta_db.mean().reindex(labels)
        rows.append(dict(condition=condition,AF3_db=float(values.AF3),AF4_db=float(values.AF4),
            unregularized_midpoint_db=float(weights[0]@values),regularized_midpoint_db=float(weights[1]@values),
            regularization=float(regularization)))
    result=pd.DataFrame(rows);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'af3_af4_midpoint_check.csv',index=False)
    print('AF3--AF4 display diagnostic: unchanged 16-participant group means; midpoint is interpolated, not measured.')
    print(result.to_string(index=False))
    return result


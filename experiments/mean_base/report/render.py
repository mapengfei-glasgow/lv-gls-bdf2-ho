"""Render true-scale solver and prediction/error frames with PyVista, then GIFs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyvista as pv


def save_gif(frames,path,fps):
    images=[Image.fromarray(frame).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)
            for frame in frames]
    images[0].save(path,save_all=True,append_images=images[1:],duration=round(1000/fps),
                   loop=0,optimize=True,disposal=2)


def render(sample,output,stride=2,fps=20):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(sample) as z:
        X=z['coords'];cells=z['cells'];U=z['u'];P=z['prediction'];times=z['time'];case=str(z['case'])
    vtkcells=np.column_stack([np.full(len(cells),4),cells]).ravel()
    grid=pv.UnstructuredGrid(vtkcells,np.full(len(cells),pv.CellType.TETRA,np.uint8),X[:,[1,2,0]]*1000)
    surface=grid.extract_surface(algorithm='dataset_surface')
    ids=np.asarray(surface.point_data['vtkOriginalPointIds'])
    umag=np.linalg.norm(U,axis=2)*1000
    pmag=np.linalg.norm(P,axis=2)*1000
    error=np.linalg.norm(P-U,axis=2)*1000
    disp_max=max(float(umag.max()),float(pmag.max()))
    err_max=float(error.max())
    lo=np.minimum((X[None]+U).min(axis=(0,1)),(X[None]+P).min(axis=(0,1)))[[1,2,0]]*1000
    hi=np.maximum((X[None]+U).max(axis=(0,1)),(X[None]+P).max(axis=(0,1)))[[1,2,0]]*1000
    center=(lo+hi)/2;extent=float((hi-lo).max())
    steps=list(range(0,len(times),stride))
    if steps[-1]!=len(times)-1:steps.append(len(times)-1)
    for comparison in [False,True]:
        name='evaluation' if comparison else 'simulation'
        frame_dir=output/(name+'_frames');frame_dir.mkdir(exist_ok=True)
        plot=pv.Plotter(off_screen=True,shape=(1,3) if comparison else (1,1),
                        window_size=(1350,650) if comparison else (760,740))
        plot.set_background('#f7fafc')
        meshes=[];clocks=[]
        titles=['Solver','Emulator','Pointwise error'] if comparison else ['Finite-element simulation']
        for panel,title in enumerate(titles):
            plot.subplot(0,panel)
            plot.set_background('#f7fafc')
            mesh=surface.copy(deep=True)
            mesh.point_data['value']=np.zeros(len(ids))
            maximum=err_max if panel==2 else disp_max
            scalar_title='Pointwise displacement error (mm)' if panel==2 else '|u| (mm)'
            if comparison and panel<2:scalar_title=title+' '+scalar_title
            plot.add_mesh(mesh,scalars='value',clim=(0,max(maximum,1e-6)),
                          cmap='magma' if panel==2 else 'viridis',show_edges=False,
                          smooth_shading=False,ambient=.35,diffuse=.65,
                          scalar_bar_args=dict(title=scalar_title,color='#223345',
                              title_font_size=12,label_font_size=11,position_x=.12,position_y=.035,
                              width=.76,height=.08,n_labels=4,fmt='%.1f'))
            plot.add_text(title,position='upper_edge',font_size=15,color='#223345')
            clocks.append(plot.add_text('',position=(12,(650 if comparison else 740)-115),font_size=11,color='#41576c'))
            plot.camera_position=[center+extent*np.array([1.9,1.6,.8]),center,(0,0,1)]
            plot.enable_parallel_projection();plot.camera.parallel_scale=extent*.88
            meshes.append(mesh)
        frames=[]
        for count,n in enumerate(steps):
            for panel,mesh in enumerate(meshes):
                displacement=P[n] if panel==1 else U[n]
                mesh.points=(X[ids]+displacement[ids])[:,[1,2,0]]*1000
                mesh.point_data['value']=(error[n] if panel==2 else pmag[n] if panel==1 else umag[n])[ids]
                clocks[panel].SetInput(f't = {times[n]:.3f} s\nt/T = {n/(len(times)-1):.2f}\nTrue scale (1x)')
            plot.render()
            frame=plot.screenshot(frame_dir/f'frame_{count:03d}.png',return_img=True)
            frames.append(frame)
            if count==len(steps)//2:
                Image.fromarray(frame).save(output/(name+'_snapshot.png'))
        plot.close()
        save_gif(frames,output/(name+'.gif'),fps)
        with Image.open(output/(name+'.gif')) as gif:
            assert gif.n_frames==len(steps)
        print(f'{name}: {len(frames)} PyVista frames -> {output/(name+".gif")}',flush=True)
    (output/'animation.json').write_text(json.dumps(dict(case=case,frames=len(steps),fps=fps,
        cycle_duration_s=float(times[-1]),deformation_scale=1,
        displacement_color_max_mm=disp_max,error_color_max_mm=err_max,
        renderer='PyVista offscreen VTK',
        note='GIFs slow the first prescribed cycle for inspection; the end state need not equal the initially undeformed state.'),indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sample',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--stride',type=int,default=2)
    ap.add_argument('--fps',type=int,default=20)
    a=ap.parse_args();render(a.sample,a.output,a.stride,a.fps)

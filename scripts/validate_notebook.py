"""Execute the notebook from a clean namespace; capture text and plots without Jupyter."""
import base64, contextlib, io, json, os, sys, traceback
from pathlib import Path
os.environ.setdefault('TRANSDUCER_PROFILE','smoke')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/transducers-matplotlib')
path=Path('notebooks/hidden_channels_prediction_coordination.ipynb')
nb=json.loads(path.read_text())
assert nb['nbformat']==4
scope={'__name__':'__main__'}
count=0
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code': continue
    compile(cell['source'],f'cell-{i}','exec')
    count+=1;cell['execution_count']=count;cell['outputs']=[]
    output=io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if 'matplotlib.pyplot' in sys.modules:
                import matplotlib.pyplot as plt
                def show(*a,**kw):
                    for n in plt.get_fignums():
                        fig=plt.figure(n);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120,bbox_inches='tight')
                        cell['outputs'].append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(buf.getvalue()).decode(),'text/plain':['Matplotlib figure']}})
                        out=Path('validation');out.mkdir(exist_ok=True)
                        (out/f'cell_{i}_figure_{n}.png').write_bytes(buf.getvalue())
                    plt.close('all')
                plt.show=show
            exec(cell['source'],scope)
    except Exception:
        print(output.getvalue());traceback.print_exc();sys.exit(1)
    if output.getvalue():cell['outputs'].insert(0,{'output_type':'stream','name':'stdout','text':output.getvalue()})
    print(f'PASS cell {i}', output.getvalue()[:350],flush=True)
assert scope['RUN_TRAINING'] and len(scope['MODELS'])==2*len(scope['CFG']['seeds'])
assert len(scope['PROBES'])==len(scope['MODELS'])*(scope['CFG']['layers']+1)
import numpy as np
for r in scope['RESULTS']:
    assert np.isfinite(r['test_CE']) and r['expected_excess_CE_KL']>=-1e-8
for r in scope['PROBES']:
    assert np.isfinite(r['joint_MSE']) and r['decoded_parity_regret']>=-1e-8
# Save the executed notebook with conspicuous validation provenance.
nb['metadata']['validation']={'profile':scope['PROFILE'],'scope':'Local execution test; not a converged neural experiment'}
nb['cells'] = [c for c in nb['cells'] if c.get('id') != 'validation-note']
nb['cells'].insert(1, {'cell_type':'markdown','id':'validation-note','metadata':{},'source':f'**Included run:** all code cells executed locally with `{scope["PROFILE"]}`: {scope["CFG"]["steps"]} training steps per interface, {len(scope["CFG"]["seeds"])} seed(s), {scope["DEVICE"]}. These are exploratory outputs, not a converged or replicated study. Colab execution itself has not been tested.\n'})
path.write_text(json.dumps(nb,indent=1))
print('SUCCESS: all notebook cells executed; output captured in',path)

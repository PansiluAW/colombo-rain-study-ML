"""Colombo rain notebook: a small interface for the saved Regularised QDA study."""
from datetime import date,timedelta
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.data import ROOT, FEATURES, LABELS
from src.predict import INPUTS, LIMITS, load_model, make_input, assess

st.set_page_config(page_title='Colombo, tomorrow.',page_icon='☂',layout='wide')
st.markdown('''<style>
.stApp{background-image:linear-gradient(#4a452d09 1px,transparent 1px);background-size:100% 30px}
.block-container{max-width:1130px;padding-top:2rem;padding-bottom:2rem}
header[data-testid="stHeader"]{background:transparent}
h1{font-family:Georgia,"Times New Roman",serif!important;font-size:3.1rem!important;font-weight:400!important;letter-spacing:-1.5px!important;margin-bottom:0!important}
h2,h3{font-family:monospace!important;font-weight:500!important;letter-spacing:0!important}
h3{font-size:1rem!important}
[data-testid="stForm"]{border:0;padding:0}
[data-baseweb="input"],[data-baseweb="select"]>div{border-radius:0!important}
button{border-radius:0!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:2rem;border-bottom:1px solid #7b7a6e}
[data-testid="stTabs"] [data-baseweb="tab"]{font-size:13px}
.station-line{font-family:monospace;font-size:11px;letter-spacing:1.1px;border-top:2px solid #292820;border-bottom:1px solid #aaa797;padding:8px 0;margin:10px 0 18px}
.result-word{font-family:monospace;font-size:76px;line-height:1.1;letter-spacing:-6px;margin-top:12px;margin-bottom:6px;color:#30597a}
.probability{font-family:Georgia,serif;font-size:29px;margin:0 0 12px}
.pencil{font-family:Georgia,serif;font-size:15px;font-style:italic;color:#6d6556;line-height:1.5}
.rule{border-bottom:1px solid #aaa797;margin:14px 0}
.footer-note{border-top:1px solid #aaa797;padding-top:12px;font-size:10px;color:#777568;margin-top:26px}
[data-testid="stMetric"]{border:0!important;background:transparent!important}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def model_resource():
    return load_model()

@st.cache_data
def records():
    return (json.loads((ROOT/'results/metrics.json').read_text()),
            pd.read_csv(ROOT/'data/model_dataset.csv'),
            pd.read_csv(ROOT/'results/shap_importance.csv'))

model=model_resource()
summary,data,importance=records()
threshold=summary['test']['Regularised QDA']['threshold']
examples=summary['examples']
options=[f'{date.fromisoformat(ex["date"]):%d %b %Y} / {ex["name"].lower()}' for ex in examples]
options.append('My own entry')

def select_example():
    choice=st.session_state.get('example',options[0])
    if choice==options[-1]:return
    ex=examples[options.index(choice)]
    for key,value in zip(INPUTS,ex['features']):
        st.session_state[key]=float(value)
    st.session_state['target_date']=date.fromisoformat(ex['date'])

if 'target_date' not in st.session_state:
    select_example()

st.title('Colombo, tomorrow.')
st.markdown('<div class="station-line">RAIN NOTEBOOK / 06.9271 N, 79.8612 E / QDA STUDY / 269220E</div>',unsafe_allow_html=True)
notebook,results,dataset=st.tabs(['01  Make an entry','02  Test results','03  About the data'])

def shap_figure(values, labels, top=6):
    indices=np.argsort(np.abs(values))[-top:]
    fig,ax=plt.subplots(figsize=(6,3.0))
    fig.patch.set_alpha(0);ax.set_facecolor('none')
    ax.barh(np.array(labels)[indices],np.array(values)[indices],
            color=['#30597a' if values[i]>=0 else '#a06842' for i in indices],height=.58)
    ax.axvline(0,color='#8c8a7c',linewidth=.7)
    ax.tick_params(axis='both',labelsize=8,length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xlabel('Contribution to the raw wet-day score (log odds)',fontsize=8)
    fig.tight_layout()
    return fig

with notebook:
    left,right=st.columns([1.16,1],gap='large')
    with left:
        st.selectbox('Start with a recorded day, or make your own entry',options,key='example',on_change=select_example)
        with st.form('weather'):
            st.date_input('Date to predict',key='target_date')
            st.caption('Use the completed day before this date for the readings below.')
            a,b=st.columns(2)
            fields=[('Rain, preceding day (mm)',0,1),('Mean humidity (%)',1,1),
                    ('Maximum temperature (°C)',2,.1),('Minimum temperature (°C)',3,.1),
                    ('Maximum wind (km/h)',4,.1),('Mean pressure (hPa)',5,.1),
                    ('3-day mean rain (mm/day)',6,.1),('7-day mean rain (mm/day)',7,.1),
                    ('Pressure change (hPa)',8,.1)]
            for index,(label,i,step) in enumerate(fields):
                with a if index%2==0 else b:
                    low,high=LIMITS[i]
                    st.number_input(label,min_value=float(low),max_value=float(high),step=float(step),format='%.2f',key=INPUTS[i])
            st.caption('Rain means include the preceding day. Pressure change = that day’s mean minus the day before it.')
            st.form_submit_button('Work out the next day →',width='stretch')
        st.markdown('<p class="pencil">A wet day means at least 1 mm of rain. The model uses 11 inputs: the nine readings here and two seasonal values from the date.</p>',unsafe_allow_html=True)
    with right:
        try:
            frame=make_input(st.session_state['target_date'],st.session_state)
            prediction=assess(model,frame,threshold)
        except (ValueError,TypeError) as error:
            st.error(str(error));prediction=None
        if prediction:
            st.caption(f'ENTRY FOR {st.session_state["target_date"]:%d %b %Y}')
            st.markdown(f'<div class="result-word">{prediction["label"]}</div><div class="probability">{prediction["probability"]:.1%} model probability</div>',unsafe_allow_html=True)
            st.caption(f'Wet-day cutoff: {threshold:.0%}. Probabilities have not been independently calibrated.')
            matched=next((ex for ex in examples if ex['date']==st.session_state['target_date'].isoformat()
                and np.allclose(frame.iloc[0].to_numpy(),ex['features'],atol=1e-10,rtol=0)),None)
            if matched:
                st.write(f'Recorded next-day rain: **{matched["actual_rain"]:.1f} mm** / {"wet" if matched["actual"] else "dry"}.')
            st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
            st.markdown('### What moved the score?')
            st.caption('Native interventional SHAP, calculated for this entry. Blue pushes towards wet; brown pushes towards dry.')
            figure=shap_figure(prediction['shap'],LABELS)
            st.pyplot(figure,clear_figure=True);plt.close(figure)
            train=data[data.target_date<'2022-01-01']
            outside=[LABELS[i] for i,key in enumerate(INPUTS) if frame.iloc[0][key]<train[key].min() or frame.iloc[0][key]>train[key].max()]
            if outside:st.warning('Outside the training range: '+', '.join(outside)+'.')
            with st.expander('All eleven contributions'):
                st.dataframe(pd.DataFrame({'Input':LABELS,'Value':frame.iloc[0].to_numpy(),
                    'SHAP (log odds)':prediction['shap']}),hide_index=True,width='stretch')
                st.write(f'Base {prediction["base"]:.3f} + contributions {sum(prediction["shap"]):.3f} = raw score {prediction["raw"]:.3f}.')
                st.caption('The logistic transform converts the raw score to a probability. Exact interventional SHAP explains the quadratic score against the training background. Replacements can break correlations and the sine/cosine relationship; these are model associations, not causal effects.')
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    st.caption('Research note: these inputs come from delayed ERA5 reanalysis. This checks historical rain-day patterns; it is not a live forecast or a flood warning.')

with results:
    st.markdown('### Leave the last two years out')
    st.write('The model was fitted on 2014–2021, selected on 2022–2023 and tested on 2024–2025. The test set contains 731 days: 556 wet and 175 dry.')
    table=[]
    for name,metrics in summary['test'].items():
        table.append({'Model':name,'ROC AUC':metrics['auc'],'Accuracy':metrics['accuracy'],
            'Balanced accuracy':metrics['balanced_accuracy'],'F1':metrics['f1'],'Brier loss':metrics['brier']})
    st.dataframe(pd.DataFrame(table).set_index('Model').style.format('{:.3f}'),width='stretch')
    st.write('Logistic regression gives higher AUC and lower Brier loss than QDA. Persistence gives higher accuracy and F1; monthly climatology gives higher balanced accuracy.')
    a,b=st.columns(2)
    with a:
        m=summary['test']['Regularised QDA']
        st.markdown('### Where the predictions went wrong')
        st.table(pd.DataFrame([[m['tn'],m['fp']],[m['fn'],m['tp']]],index=['Actual dry','Actual wet'],columns=['Predicted dry','Predicted wet']))
        st.caption(f'{m["fn"]} wet days were missed. There were {m["fp"]} false wet alerts. Approximate 95% block-bootstrap AUC interval: {summary["auc_ci_95"][0]:.3f}–{summary["auc_ci_95"][1]:.3f}.')
    with b:
        st.markdown('### Inputs used most often')
        fig,ax=plt.subplots(figsize=(6,3.2));fig.patch.set_alpha(0);ax.set_facecolor('none')
        g=importance.head(6).iloc[::-1]
        ax.barh(g.label,g.mean_abs_shap,color='#30597a',height=.55)
        ax.set_xlabel('Mean absolute interventional SHAP (log odds)',fontsize=8);ax.tick_params(labelsize=8,length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        fig.tight_layout();st.pyplot(fig,clear_figure=True);plt.close(fig)
        st.caption('240 evenly spaced test days; all training rows form the SHAP background. Interventional replacements can break relationships between inputs.')
    with st.expander('Training choices'):
        st.write(f'Quadratic Discriminant Analysis; covariance regularisation {summary["selected_params"]["reg_param"]}. Eight settings were compared on validation AUC. Rainfall inputs use log1p, then all inputs are standardised using training rows only. The {threshold:.2f} cutoff was selected on validation balanced accuracy. No deep learning is used.')
        st.dataframe(pd.DataFrame(summary['candidates']),hide_index=True,width='stretch')

with dataset:
    st.markdown('### A compiled local dataset')
    st.write('4,383 daily records were downloaded from Open-Meteo using the ECMWF ERA5 model, covering 1 January 2014 to 31 December 2025 in the Asia/Colombo time zone. Removing incomplete seven-day windows and the last unavailable target left 4,376 modelling rows.')
    st.write('Requested point: 6.9271 N, 79.8612 E. Returned grid location: 7.0 N, 80.0 E. These are gridded estimates for the Colombo area, rather than measurements at a particular weather station.')
    st.dataframe(data[['observation_date','target_date','rain_today','humidity_today','target_rain_mm','wet_tomorrow']].head(12),hide_index=True,width='stretch')
    st.download_button('Download the daily data (CSV)',data=(ROOT/'data/colombo_daily.csv').read_bytes(),file_name='colombo_daily.csv',mime='text/csv')
    st.markdown('### Limits worth keeping in mind')
    st.write('A single grid cell misses local variation. Wet days are common in this extract. The one-millimetre boundary creates a sharp distinction between small rainfall amounts. The data arrive after the observation day, so this experiment does not establish operational forecast skill.')
    st.write('The dataset contains no personal information. Wider use would require station checks, more locations, calibration and testing with inputs actually available at forecast time.')
    st.markdown('Data: [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) / ECMWF ERA5 / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Daily JSON was converted to CSV and lagged, rolling and seasonal predictors were derived.')

st.markdown('<div class="footer-note">Pansilu Wijesiri · 269220E · MSc in Artificial Intelligence<br>Colombo rain-day classification / QDA / 2014–2025</div>',unsafe_allow_html=True)

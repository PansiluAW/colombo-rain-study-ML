from src.data import ROOT, make_frame

if __name__ == '__main__':
    frame=make_frame()
    frame.to_csv(ROOT/'data/model_dataset.csv',index=False)
    print(f'Prepared {len(frame)} supervised rows and 11 predictors.')

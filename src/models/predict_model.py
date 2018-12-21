from abs_module import *
import os
if __name__=='__main__':
    model=absenteeism(r'model.pickle',r'scaler.pickle')
    model.load_and_process_data(os.path.join(os.path.pardir,os.path.pardir,'data','raw','Absenteeism_at_work.csv'))
    print(model.prediction_with_inputs())

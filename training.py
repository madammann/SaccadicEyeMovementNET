def write_performance(model,loss,accuracy,epoch):
    try:
        df = pd.read_csv('./weights/index.csv')
        df.append(model.name,epoch,accuracy,loss,columns=['Model','Epoch','Accuracy','Loss','Classes'])
        df.to_csv('./weights/index.csv',index=False)
    except Exception as e:
        print(e)
        print('Could not write model performance to index file.')
        
def do_epoch(sn_eye,sn_classifier,dataset,epoch):
    pass
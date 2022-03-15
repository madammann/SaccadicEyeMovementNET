def train():
    pass

def write_performance(model,loss,accuracy,epoch):
    try:
        df = pd.read_csv('./weights/index.csv')
        df.append(model.nameinfo[0],model.nameinfo[1],model.nameinfo[2],epoch,accuracy,loss,columns=['Model','Part','Id','Epoch','Accuracy','Loss','Classes'])
        df.to_csv('./weights/index.csv',index=False)
    except Exception as e:
        print(e)
        print('Could not write model performance to index file.')
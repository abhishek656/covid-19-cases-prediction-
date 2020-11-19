#importing the modules
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

#intialising the Falsk , App get created
app = Flask(__name__)

#here first homepage will get display in website
@app.route('/')
def home():
    return render_template('index.html')






#here in frontend when we click predict , then it will redirect to "/predict"
#1) covid19 prediction for worldwide confirmed cases
@app.route("/predict",methods=['GET','POST'])
#creating a function
def  predict():
    if request.method == 'POST':
        
        #here the input data get stored in date with help of request.form
        date=int(request.form['date'])

        
        
        #making values into array
        pred_args_arr=np.array(date)
        
        #converting from 1 dimention to 2 dimentional
        pred_args_arr=pred_args_arr.reshape(-1,1)

        #loading the pickle file  
        model=pickle.load(open("model.pkl","rb"))
        x=np.arange(len(model))  #it arange based on array index assumend number of days as independent variable on x-axies
        x=x.reshape(-1,1) 
        y=model.values

        from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
        poly = PolynomialFeatures(degree=4)  #Applying degree of 4
        X= poly.fit_transform(x)
        


        from sklearn.linear_model import LinearRegression #linaer regression 
        regressor = LinearRegression() # storing linearregression object in regressor1
        regressor.fit(X, y)
        
        #predict the output
        model_prediction=regressor.predict(poly.transform(pred_args_arr)) 

        #it is use to eliminate the array
        pred_args_arr=round(int(pred_args_arr),2) 
        model_prediction=round(int(model_prediction),2) 

    #displaying the the backend output by replacing in frontend prediction word
    return  render_template('index.html', prediction='Predicted Confirmed cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr,model_prediction))












#covid19 prediction for russia confirmed cases

@app.route("/predict_russia_confirmed",methods=['GET','POST'])
#creating a function
def  predict_russia_confirmed():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date1=int(request.form['date1'])

                  #making values into array
                 pred_args_arr1=np.array(date1)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr1=pred_args_arr1.reshape(-1,1)

                   #loading the pickle file  
                 model1=pickle.load(open("modelrussiaconfirmed.pkl","rb"))
                 x1=np.arange(len(model1))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x1=x1.reshape(-1,1) 
                 y1=model1.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly1 = PolynomialFeatures(degree=7)  #Applying degree of 7
                 X1= poly1.fit_transform(x1)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor1 = LinearRegression() # storing linearregression object in regressor1
                 regressor1.fit(X1, y1)

                  #predict the output
                 model_prediction1=regressor1.predict(poly1.transform(pred_args_arr1)) 

                 #it is use to eliminate the array
                 pred_args_arr1=round(int(pred_args_arr1),2) 
                 model_prediction1=round(int(model_prediction1),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction1='Pridicted  russia Confirmed cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr1,model_prediction1))








#covid19 prediction for russia Active cases

@app.route("/predict_russia_Active",methods=['GET','POST'])
#creating a function
def  predict_russia_Active():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date2=int(request.form['date2'])

                  #making values into array
                 pred_args_arr2=np.array(date2)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr2=pred_args_arr2.reshape(-1,1)

                   #loading the pickle file  
                 model2=pickle.load(open("modelrussiaActive.pkl","rb"))
                 x2=np.arange(len(model2))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x2=x2.reshape(-1,1) 
                 y2=model2.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly2 = PolynomialFeatures(degree=8) 
                 X2= poly2.fit_transform(x2)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor2 = LinearRegression() # storing linearregression object in regressor1
                 regressor2.fit(X2, y2)

                  #predict the output
                 model_prediction2=regressor2.predict(poly2.transform(pred_args_arr2)) 

                 #it is use to eliminate the array
                 pred_args_arr2=round(int(pred_args_arr2),2) 
                 model_prediction2=round(int(model_prediction2),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction2='Predicted  russia Active cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr2,model_prediction2))








#covid19 prediction for russia  deaths cases

@app.route("/predict_russia_deaths",methods=['GET','POST'])
#creating a function
def  predict_russia_deaths():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date3=int(request.form['date3'])

                  #making values into array
                 pred_args_arr3=np.array(date3)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr3=pred_args_arr3.reshape(-1,1)

                   #loading the pickle file  
                 model3=pickle.load(open("modelrussiadeaths.pkl","rb"))
                 x3=np.arange(len(model3))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x3=x3.reshape(-1,1) 
                 y3=model3.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly3 = PolynomialFeatures(degree=7) 
                 X3= poly3.fit_transform(x3)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor3 = LinearRegression() # storing linearregression object in regressor1
                 regressor3.fit(X3, y3)

                  #predict the output
                 model_prediction3=regressor3.predict(poly3.transform(pred_args_arr3)) 

                 #it is use to eliminate the array
                 pred_args_arr3=round(int(pred_args_arr3),2) 
                 model_prediction3=round(int(model_prediction3),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction3='Predicted  russia deaths cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr3,model_prediction3))

        
 






 #covid19 prediction for russia  New cases

@app.route("/predict_russia_New_cases",methods=['GET','POST'])
#creating a function
def  predict_russia_New_cases():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date4=int(request.form['date4'])

                  #making values into array
                 pred_args_arr4=np.array(date4)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr4=pred_args_arr4.reshape(-1,1)

                   #loading the pickle file  
                 model4=pickle.load(open("modelrussiaNewcases.pkl","rb"))
                 x4=np.arange(len(model4))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x4=x4.reshape(-1,1) 
                 y4=model4.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly4 = PolynomialFeatures(degree=8) 
                 X4= poly4.fit_transform(x4)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor4 = LinearRegression() # storing linearregression object in regressor1
                 regressor4.fit(X4, y4)

                  #predict the output
                 model_prediction4=regressor4.predict(poly4.transform(pred_args_arr4)) 

                 #it is use to eliminate the array
                 pred_args_arr4=round(int(pred_args_arr4),2) 
                 model_prediction4=round(int(model_prediction4),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction4='Predicted  russia New cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr4,model_prediction4))






#covid19 prediction for russia  New deaths

@app.route("/predict_russia_New_deaths",methods=['GET','POST'])
#creating a function
def  predict_russia_New_deaths():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date5=int(request.form['date5'])

                  #making values into array
                 pred_args_arr5=np.array(date5)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr5=pred_args_arr5.reshape(-1,1)

                   #loading the pickle file  
                 model5=pickle.load(open("modelrussiaNewcases.pkl","rb"))
                 x5=np.arange(len(model5))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x5=x5.reshape(-1,1) 
                 y5=model5.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly5 = PolynomialFeatures(degree=8) 
                 X5= poly5.fit_transform(x5)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor5 = LinearRegression() # storing linearregression object in regressor1
                 regressor5.fit(X5, y5)

                  #predict the output
                 model_prediction5=regressor5.predict(poly5.transform(pred_args_arr5)) 

                 #it is use to eliminate the array
                 pred_args_arr5=round(int(pred_args_arr5),2) 
                 model_prediction5=round(int(model_prediction5),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction5='Predicted  russia New deaths cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr5,model_prediction5))







#covid19 prediction for russia  New Recovered

@app.route("/predict_russia_New_Recovered",methods=['GET','POST'])
#creating a function
def  predict_russia_New_Recovered():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date6=int(request.form['date6'])

                  #making values into array
                 pred_args_arr6=np.array(date6)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr6=pred_args_arr6.reshape(-1,1)

                   #loading the pickle file  
                 model6=pickle.load(open("modelrussiaNewrecovered.pkl","rb"))
                 x6=np.arange(len(model6))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x6=x6.reshape(-1,1) 
                 y6=model6.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly6 = PolynomialFeatures(degree=14) 
                 X6= poly6.fit_transform(x6)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor6 = LinearRegression() # storing linearregression object in regressor1
                 regressor6.fit(X6, y6)

                  #predict the output
                 model_prediction6=regressor6.predict(poly6.transform(pred_args_arr6)) 

                 #it is use to eliminate the array
                 pred_args_arr6=round(int(pred_args_arr6),2) 
                 model_prediction6=round(int(model_prediction6),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction6='Predicted  russia New Recovered cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr6,model_prediction6))


#covid19 prediction for russia  Recovered

@app.route("/predict_russia_Recovered",methods=['GET','POST'])
#creating a function
def  predict_russia_Recovered():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date7=int(request.form['date7'])

                  #making values into array
                 pred_args_arr7=np.array(date7)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr7=pred_args_arr7.reshape(-1,1)

                   #loading the pickle file  
                 model7=pickle.load(open("modelrussiaRecovered.pkl","rb"))
                 x7=np.arange(len(model7))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x7=x7.reshape(-1,1) 
                 y7=model7.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly7 = PolynomialFeatures(degree=6) 
                 X7= poly7.fit_transform(x7)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor7 = LinearRegression() # storing linearregression object in regressor1
                 regressor7.fit(X7, y7)

                  #predict the output
                 model_prediction7=regressor7.predict(poly7.transform(pred_args_arr7)) 

                 #it is use to eliminate the array
                 pred_args_arr7=round(int(pred_args_arr7),2) 
                 model_prediction7=round(int(model_prediction7),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction7='Predicted  russia Recovered cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr7,model_prediction7))




#covid19 prediction for iran confirmed cases

@app.route("/predict_Iran_confirmed",methods=['GET','POST'])
#creating a function
def  predict_Iran_confirmed():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date8=int(request.form['date8'])

                  #making values into array
                 pred_args_arr8=np.array(date8)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr8=pred_args_arr8.reshape(-1,1)

                   #loading the pickle file  
                 model8=pickle.load(open("modelIranconfirmed.pkl","rb"))
                 x8=np.arange(len(model8))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x8=x8.reshape(-1,1) 
                 y8=model8.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly8 = PolynomialFeatures(degree=7) 
                 X8= poly8.fit_transform(x8)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor8 = LinearRegression() # storing linearregression object in regressor1
                 regressor8.fit(X8, y8)

                  #predict the output
                 model_prediction8=regressor8.predict(poly8.transform(pred_args_arr8)) 

                 #it is use to eliminate the array
                 pred_args_arr8=round(int(pred_args_arr8),2) 
                 model_prediction8=round(int(model_prediction8),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction8='Predicted  Iran confirmed cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr8,model_prediction8))







#covid19 prediction for uttar pradesh confirmed cases

@app.route("/predict_up_confirmed",methods=['GET','POST'])
#creating a function
def  predict_up_confirmed():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date9=int(request.form['date9'])

                  #making values into array
                 pred_args_arr9=np.array(date9)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr9=pred_args_arr9.reshape(-1,1)

                   #loading the pickle file  
                 model9=pickle.load(open("modelupconfirmed.pkl","rb"))
                 x9=np.arange(len(model9))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x9=x9.reshape(-1,1) 
                 y9=model9.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly9 = PolynomialFeatures(degree=7) 
                 X9= poly9.fit_transform(x9)

                 from sklearn.linear_model import LinearRegression #linaer regression 
                 regressor9 = LinearRegression() # storing linearregression object in regressor1
                 regressor9.fit(X9, y9)

                  #predict the output
                 model_prediction9=regressor9.predict(poly9.transform(pred_args_arr9)) 

                 #it is use to eliminate the array
                 pred_args_arr9=round(int(pred_args_arr9),2) 
                 model_prediction9=round(int(model_prediction9),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction9 ='Predicted  uttar pradesh confirmed cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr9,model_prediction9))






#covid19 prediction for uttar pradesh  Active  cases

@app.route("/predict_up_Active",methods=['GET','POST'])
#creating a function
def  predict_up_Active():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date10=int(request.form['date10'])

                  #making values into array
                 pred_args_arr10=np.array(date10)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr10=pred_args_arr10.reshape(-1,1)

                   #loading the pickle file  
                 model10=pickle.load(open("modelupActive.pkl","rb"))
                 x10=np.arange(len(model10))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x10=x10.reshape(-1,1) 
                 y10=model10.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly10 = PolynomialFeatures(degree=7) 
                 X10= poly10.fit_transform(x10)

                 from sklearn.linear_model import LinearRegression #linaer regression
                 regressor10 = LinearRegression() # storing linearregression object in regressor1
                 regressor10.fit(X10, y10)

                  #predict the output
                 model_prediction10=regressor10.predict(poly10.transform(pred_args_arr10)) 

                 #it is use to eliminate the array
                 pred_args_arr10=round(int(pred_args_arr10),2) 
                 model_prediction10=round(int(model_prediction10),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction10 ='Predicted  uttar pradesh Active cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr10,model_prediction10))







#covid19 prediction for uttar pradesh  deaths  cases

@app.route("/predict_up_deaths",methods=['GET','POST'])
#creating a function
def  predict_up_deaths():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date11=int(request.form['date11'])

                  #making values into array
                 pred_args_arr11=np.array(date11)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr11=pred_args_arr11.reshape(-1,1)

                   #loading the pickle file  
                 model11=pickle.load(open("modelupdeaths.pkl","rb"))
                 x11=np.arange(len(model11))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x11=x11.reshape(-1,1) 
                 y11=model11.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly11 = PolynomialFeatures(degree=7) 
                 X11= poly11.fit_transform(x11)

                 from sklearn.linear_model import LinearRegression #linaer regression
                 regressor11 = LinearRegression() # storing linearregression object in regressor1
                 regressor11.fit(X11, y11)

                  #predict the output
                 model_prediction11=regressor11.predict(poly11.transform(pred_args_arr11)) 

                 #it is use to eliminate the array
                 pred_args_arr11=round(int(pred_args_arr11),2) 
                 model_prediction11=round(int(model_prediction11),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction11 ='Predicted  uttar pradesh deaths cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr11,model_prediction11))





#covid19 prediction for uttar pradesh  Recovered  cases

@app.route("/predict_up_Recovered",methods=['GET','POST'])
#creating a function
def  predict_up_Recovered():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date12=int(request.form['date12'])

                  #making values into array
                 pred_args_arr12=np.array(date12)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr12=pred_args_arr12.reshape(-1,1)

                   #loading the pickle file  
                 model12=pickle.load(open("modelupRecovered .pkl","rb"))
                 x12=np.arange(len(model12))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x12=x12.reshape(-1,1) 
                 y12=model12.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly12 = PolynomialFeatures(degree=7) 
                 X12= poly12.fit_transform(x12)

                 from sklearn.linear_model import LinearRegression #linaer regression
                 regressor12 = LinearRegression() # storing linearregression object in regressor1
                 regressor12.fit(X12, y12)

                  #predict the output
                 model_prediction12=regressor12.predict(poly12.transform(pred_args_arr12)) 

                 #it is use to eliminate the array
                 pred_args_arr12=round(int(pred_args_arr12),2) 
                 model_prediction12=round(int(model_prediction12),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction12 ='Predicted  uttar pradesh  Recovered cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr12,model_prediction12))






#covid19 prediction for uttar pradesh  New cases

@app.route("/predict_up_new_cases",methods=['GET','POST'])
#creating a function
def  predict_up_new_cases():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date13=int(request.form['date13'])

                  #making values into array
                 pred_args_arr13=np.array(date13)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr13=pred_args_arr13.reshape(-1,1)

                   #loading the pickle file  
                 model13=pickle.load(open("modelupNewcases.pkl","rb"))
                 x13=np.arange(len(model13))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x13=x13.reshape(-1,1) 
                 y13=model13.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly13= PolynomialFeatures(degree=8) 
                 X13= poly13.fit_transform(x13)

                 from sklearn.linear_model import LinearRegression #linaer regression
                 regressor13 = LinearRegression() # storing linearregression object in regressor1
                 regressor13.fit(X13, y13)

                  #predict the output
                 model_prediction13=regressor13.predict(poly13.transform(pred_args_arr13)) 

                 #it is use to eliminate the array
                 pred_args_arr13=round(int(pred_args_arr13),2) 
                 model_prediction13=round(int(model_prediction13),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction13='Predicted  uttar pradesh  new  cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr13,model_prediction13))




#covid19 prediction for uttar pradesh  New Recovered  cases

@app.route("/predict_up_New_Recovered",methods=['GET','POST'])
#creating a function
def  predict_up_New_Recovered():
            if request.method == 'POST':
                 #here the input data get stored in date with help of request.form
                 date14=int(request.form['date14'])

                  #making values into array
                 pred_args_arr14=np.array(date14)

                  #converting from 1 dimention to 2 dimentional
                 pred_args_arr14=pred_args_arr14.reshape(-1,1)

                   #loading the pickle file  
                 model14=pickle.load(open("modelupNewRecovered.pkl","rb"))
                 x14=np.arange(len(model14))  #it arange based on array index assumend number of days as independent variable on x-axies
                 x14=x14.reshape(-1,1) 
                 y14=model14.values

                 from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
                 poly14 = PolynomialFeatures(degree=10) 
                 X14= poly14.fit_transform(x14)

                 from sklearn.linear_model import LinearRegression #linaer regression
                 regressor14 = LinearRegression() # storing linearregression object in regressor1
                 regressor14.fit(X14, y14)

                  #predict the output
                 model_prediction14=regressor14.predict(poly14.transform(pred_args_arr14)) 

                 #it is use to eliminate the array
                 pred_args_arr14=round(int(pred_args_arr14),2) 
                 model_prediction14=round(int(model_prediction14),2) 

                  #displaying the the backend output by replacing in frontend prediction word
            return  render_template('index.html', prediction14 ='Predicted  uttar pradesh New  Recovered cases for {}th day by applying polynomial regression algorithm is [{}]'.format(pred_args_arr14,model_prediction14))






   
#main function         
if __name__ == "__main__":
     app.run(debug=True)
   
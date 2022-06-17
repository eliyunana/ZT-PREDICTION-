from tkinter import *
import numpy as np
from tkinter import messagebox 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import customtkinter
customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    WIDTH =1000
    HEIGHT =620
    def __init__(self):
        super().__init__()
        self.title("ZT_prediction_details")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        #self.resizable(False,False)
        self.iconbitmap('zticon.ico')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        self.text_f=customtkinter.CTkLabel(master=self,text='ZT PREDICTOR',text_font=("Elephant",50),anchor='center')
        self.text_f.grid(row=0, column=1, sticky="nswe", padx=20)
        self.frame=customtkinter.CTkFrame(master=self,corner_radius=10)
        
        self.frame.grid(row=1, column=1, sticky="nswe", padx=20,pady=20)
        self.fram=customtkinter.CTkFrame(master=self.frame,corner_radius=20)
        self.fram.pack(fill=BOTH)
        self.fram.rowconfigure(1, weight=1)
        self.fram.columnconfigure(1, weight=1)


        #Formular Widget
        self.formular_l=customtkinter.CTkLabel(master=self.fram,text='ENTER FORMULAR:',text_font=("Roboto Medium", 25)).grid(row=0,column=0,padx=5,pady=8,sticky=W)
        self.formular_value=StringVar()
        self.formular_e=customtkinter.CTkEntry(master=self.fram,width=500,textvariable=self.formular_value)
        self.formular_e.grid(row=0,column=1,padx=5,pady=8,sticky=E)

        #Temperature widget
        self.temp_l=customtkinter.CTkLabel(master=self.fram,text='ENTER TEMPARETURE VALUE:',text_font=("Roboto Medium", 25)).grid(row=1,column=0,padx=5,pady=8,sticky=W)
        self.temp_value=StringVar()
        self.temp_e=customtkinter.CTkEntry(master=self.fram,width=500,textvariable=self.temp_value)
        self.temp_e.grid(row=1,column=1,padx=5,pady=8,sticky=E)

        #Seebeck coefficient widget
        self.seebk_c_l=customtkinter.CTkLabel(master=self.fram,text='ENTER SEEBECK COEFFICIENT:',text_font=("Roboto Medium", 25)).grid(row=2,column=0,padx=5,pady=8,sticky=W)
        self.seebk_c_value=StringVar()
        self.seebk_c_e=customtkinter.CTkEntry(master=self.fram,width=500,textvariable=self.seebk_c_value)
        self.seebk_c_e.grid(row=2,column=1,padx=5,pady=8,sticky=E)

        #Electrical conductivity widget
        self.ec_l=customtkinter.CTkLabel(master=self.fram,text='ENTER ELECTRICAL CONDUCTIVITY:',text_font=("Roboto Medium", 25)).grid(row=3,column=0,padx=5,pady=8,sticky=W)
        self.ec_value=StringVar()
        self.ec_e=customtkinter.CTkEntry(master=self.fram,width=500,textvariable=self.ec_value)
        self.ec_e.grid(row=3,column=1,padx=5,pady=8,sticky=E) 

        #Thermal conductivity widget
        self.tc_l=customtkinter.CTkLabel(master=self.fram,text='ENTER THERMAL CONDUCTIVITY:',text_font=("Roboto Medium", 25)).grid(row=4,column=0,padx=5,pady=8,sticky=W)
        self.tc_value=StringVar()
        self.tc_e=customtkinter.CTkEntry(master=self.fram,width=500,textvariable=self.tc_value)
        self.tc_e.grid(row=4,column=1,padx=5,pady=8,sticky=E)

        #power factor widget
        self.power_fact_l=customtkinter.CTkLabel(master=self.fram,text='ENTER POWER FACTOR VALUE:',text_font=("Roboto Medium", 25)).grid(row=5,column=0,padx=5,pady=8,sticky=W)
        self.power_fact_value=StringVar()
        self.power_fact_e=customtkinter.CTkEntry(master=self.fram,width=500,textvariable=self.power_fact_value)
        self.power_fact_e.grid(row=5,column=1,padx=5,pady=8,sticky=E) 


        self.predict=customtkinter.CTkButton(master=self.fram,text='VIEW ZT PREDICTED',command=self.prediction,width=500,text_font=("Roboto 15"))
        self.predict.grid(row=7,column=1,padx=5,pady=8,sticky=E)

        self.result=customtkinter.CTkLabel(master=self.fram,text='ZT VALUE:0.00',text_font=("Roboto", 25))
        self.result.grid(row=8,column=1,padx=5,pady=8,sticky=W)

        self.switch = customtkinter.CTkSwitch(master=self.fram,
                                                text="Light Mode",
                                                command=self.change_mode)
        self.switch.grid(row=7, column=0, pady=10, padx=20, sticky="w")


    def change_mode(self):
        if self.switch.get() == 0:
            customtkinter.set_appearance_mode("dark")
        else:
            customtkinter.set_appearance_mode("light")


         
    def prediction(self):
        try:
            self.ZT_prediction_details=np.array([[self.formular_value.get(),float(self.temp_value.get()),float(self.seebk_c_value.get()),float(self.ec_value.get()),float(self.tc_value.get()),float(self.power_fact_value.get())]])
            

            le=LabelEncoder()
            self.ZT_prediction_details[:,0]=le.fit_transform(self.ZT_prediction_details[:,0])
            
            #data
            self.data=pd.read_csv('dataset/ZT_filtered.csv')
            self.x=self.data.iloc[:,:6].values
            self.y=self.data.iloc[:,6].values
            self.x[:,0]=le.fit_transform(self.x[:,0])
            #training
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.20,random_state=0)
 
            #making predictions
            self.gbr=GradientBoostingRegressor(criterion='squared_error', n_estimators=1800)
            self.gbr.fit(self.x_train,self.y_train)
            self.gbr.fit(self.x_train,self.y_train)
            self.y_pred=self.gbr.predict(self.ZT_prediction_details)
            messagebox.showinfo('success','wait for result')
            self.result.config(text='ZT VALUE:'+str(round(self.y_pred[0],2)))
            
            data={'Formular':[self.formular_value.get()] ,'Temperature':[float(self.temp_value.get())],'seebeck coefficient':[float(self.seebk_c_value.get())] ,'electrical conductivity':[float(self.ec_value.get())], 'thermal conductivity':[float(self.tc_value.get())],'Power factor':[float(self.power_fact_value.get())], 'ZT':[round(self.y_pred[0],2)]}
            self.df=pd.DataFrame(data,columns=['Formular','Temperature','seebeck coefficient','electrical conductivity','thermal conductivity','Power factor','ZT'])
            self.df.to_csv('ZT_prediction_energy.csv',index=False,mode='a',header=False)    
            self.clear()
        except Exception as e:
            self.clear()
            messagebox.showerror("ERROR ALERT",'please check carefully, valid numbers are required')
            self.result.config(text='ZT VALUE:0.00')
           
                
    def clear(self):
        self.formular_e.delete(0,END)
        self.temp_e.delete(0,END)
        self.seebk_c_e.delete(0,END)
        self.ec_e.delete(0,END)
        self.tc_e.delete(0,END)
        self.power_fact_e.delete(0,END)
    def on_closing(self, event=0):
        self.destroy()
      
       
        
        

if __name__=="__main__":  
    app = App()
    app.mainloop()

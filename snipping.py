# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:06:10 2022

@author: jarom
"""
import numpy as np 
import cv2
class SNIP():
    def __init__(self,path):
        self.s_x=0
        self.s_y=0
        self.e_x=0 
        self.e_y=0
        self.points=list()
        self.pt_dict=dict()
        self.matrix_x=dict()
        self.matrix_y=dict()
        self.matrix_z=dict()
        self.pictures=dict()
        self.dict_pictures=dict()
        self.GUI=dict()
        self.path=path
        self.dict_acc=dict()
        self.idx=-1
        self.images_coordinates=dict()
    
    def init_matrix(self,matrix_x,matrix_y,matrix_z,picture):
        if (picture) not in self.matrix_x:   
            self.matrix_x.update({str(picture):matrix_x})
        else:
            print("Already yet exist:  "+str(picture))
        if (picture) not in self.matrix_y:   
            self.matrix_y.update({str(picture):matrix_y})
        else:
            print("Already yet exist:  "+str(picture))
        if (picture) not in self.matrix_z:   
            self.matrix_z.update({str(picture):matrix_z})
        else:
            print("Already yet exist:  "+str(picture+"_z"))
        
    def init_picture(self,picture,name_picture):
        if (name_picture) not in self.pictures:   
            self.pictures.update({str(name_picture):picture})
        else:
            print("Already yet exist:  "+str(name_picture))
    
    def init_image(self,image):
        self.original_image=np.copy(image)
        if len(image.shape)<3:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            
        self.image=np.copy(image) 
        self.image_first=image 
        self.mouse_pressed = False 
        self.image_to_show=image
        self.image_to_show_big = np.copy(image) 
    
    def init_images_GUI(self,image, name_of_image):
        image_path = self.path+str(image)
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)
        if isinstance(name_of_image, str):
            self.GUI.update({name_of_image:image})
            
    def mouse_callback(self,event, x, y, flags, param):     
    #    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed 
        self.x=x
        self.y=y
        self.event=event
     
        if event == cv2.EVENT_LBUTTONDOWN:  #   Lave tlacidlo mysi         
            
            self.mouse_pressed = True         
            if self.mouse_pressed_idx==0:
                self.s_y_m=self.s_y
                self.s_x_m=self.s_x
                self.mouse_pressed_idx+=1             
            self.s_x, self.s_y = x, y   
            print("LBUTTON DOWN")
            print(str(self.s_x)+" "+str(self.s_y))
            self.image_to_show = np.copy(self.image) 
     
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pressed_idx=0
            
            if self.mouse_pressed:             
                self.image_to_show = np.copy(self.image)  
                image_to_show_mean=np.mean(self.image_to_show)
                if image_to_show_mean<123:
                    self.s_y_m=self.s_y
                    self.s_x_m=self.s_x

                    cv2.rectangle(self.image_to_show, (self.s_x, self.s_y),               
                                  (x, y), (255, 255, 255), 4) #parametra stvoruholnika
                    cv2.putText(self.image_to_show,("["+str(self.B_S_VDX+self.s_y)+", "+str(self.s_x+self.B_S_HDX)+"]"), (self.s_x, self.s_y-8),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0, 250, 0),2,cv2.LINE_AA)
                else:

                    self.s_y_m=self.s_y
                    self.s_x_m=self.s_x
                    cv2.rectangle(self.image_to_show, (self.s_x, self.s_y), (x, y), (220, 10, 10), 4) #parametra stvoruholnika
                    cv2.putText(self.image_to_show,("["+str(self.B_S_VDX+self.s_y)+", "+str(self.s_x+self.B_S_HDX)+"]"), (self.s_x, self.s_y-8),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0, 250, 0),2,cv2.LINE_AA)
                if event==cv2.EVENT_LBUTTONDOWN:
                    print("["+str(self.s_y)+", "+str(self.s_x+self.udx*1000)+"]")
     
        if event == cv2.EVENT_LBUTTONUP:   
            self.mouse_pressed = False         
            self.e_x, self.e_y = x, y
            print("LBUTTON UP")
            print(str(self.s_x)+" "+str(self.s_y))
      
            
    def moving_image(self,image):
        self.image_to_show=np.copy(self.image) 
        self.image_to_show_big = np.copy(self.image_first) 
        self.udx=0
        self.vdx=0
        #####

        shape=self.image_to_show.shape
 
        
        self.s_x = self.s_y = self.e_x = self.e_y = -1 
        self.image=self.image_to_show
        self.name_window="snipping"
        cv2.namedWindow(self.name_window) 
        cv2.setMouseCallback(self.name_window, self.mouse_callback)

        while True: 
            cv2.setMouseCallback(self.name_window, self.mouse_callback)
            cv2.imshow(self.name_window, self.image_to_show)  
            k = cv2.waitKey(1)  
            self.k=k
            self.moving()
            if self.k==ord("w") or self.k==ord("a") or self.k==ord("s") or self.k==ord("d"):
                self.moving()
                try:
                    self.image_to_show=self.image_to_show_big[self.B_S_VDX:self.B_E_VDX,self.B_S_HDX:self.B_E_HDX]
                    self.image_copy=np.copy(self.image_to_show)
                    self.image_black=np.zeros((100,self.image_to_show.shape[1],3),dtype="uint8")
                except:
                    print("Worng image to show")
                
                try:
                    self.image=np.concatenate((self.image_black,self.image_copy),axis=0)
                    cv2.putText(self.image,self.label_image, (50,90),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 220, 50),3,cv2.LINE_AA)
                except:
                    print("worng black")
                    print(self.image_black.shape)
                    print(self.image_copy.shape)
                    
                try:  
                    self.image_to_show=np.copy(self.image) 
                except:
                    print("Wrong concatenate")



        
            elif k==ord("s"):
                sur=np.asarray([self.s_y,self.s_x])
                self.points.append(sur)
                
            elif k == 27 or k == ord('q') :  
                cv2.destroyAllWindows()
                break
            self.k=" "
      
         
        cv2.destroyAllWindows()
        
    def snipping(self):  
        self.image_to_show=np.copy(self.image) 
        self.image_to_show_big = np.copy(self.image_first) 
        self.udx=0
        self.vdx=0
        #####
           
        shape=self.image_to_show.shape
 
        
        self.s_x = self.s_y = self.e_x = self.e_y = -1 
        self.image=self.image_to_show
        self.name_window="snipping"
        cv2.namedWindow(self.name_window) 
        cv2.setMouseCallback(self.name_window, self.mouse_callback)

        while True: 
            cv2.setMouseCallback(self.name_window, self.mouse_callback)
            cv2.imshow(self.name_window, self.image_to_show)  
            k = cv2.waitKey(1)  
            self.k=k
            self.moving()
            if self.k==ord("w") or self.k==ord("a") or self.k==ord("s") or self.k==ord("d"):
                self.moving()
                try:
                    self.image_to_show=self.image_to_show_big[self.B_S_VDX:self.B_E_VDX,self.B_S_HDX:self.B_E_HDX]
                    
                    self.image=np.copy(self.image_to_show)
                    self.original_image_show=self.original_image[self.B_S_VDX:self.B_E_VDX,self.B_S_HDX:self.B_E_HDX]
                
                except:
                    pass
            
            if k == ord('c'): 
                if self.s_y > self.e_y:             
                    self.s_y, self.e_y = self.e_y, self.s_y         
                if self.s_x > self.e_x:             
                    self.s_x, self.e_x = self.e_x, self.s_x 
         
                if self.e_y - self.s_y > 1 and self.e_x - self.s_x > 0:             
                    self.image = self.image_to_show[self.s_y:self.e_y, self.s_x:self.e_x] 
                    self.image_to_save = np.copy(self.original_image_show[self.s_y:self.e_y, self.s_x:self.e_x]) 
                    
                    self.image_to_show = np.copy(self.image)   
                   
            elif k==ord("s"):
                sur=np.asarray([self.s_y,self.s_x])
                self.points.append(sur)
                
            elif k == 27 or k == ord('q') :  
                cv2.destroyAllWindows()
                break
            self.k=" "
      
         
        cv2.destroyAllWindows()

               
        
    def matched_pictures(self,picture_1,picture_2,methods,name_picture_1,name_picture_2):
        img=np.copy(picture_1)
        picture_21_edit=np.copy(picture_2)
        
        if self.idx==-1:
            self.idx+=1
        
            self.init_image(picture_21_edit)
            self.snipping()
            self.template_s_x=self.s_x_m+self.udx*1000
            self.template_s_y=self.s_y_m
            
            template = self.image_to_show[3:self.image_to_show.shape[0]-3,3:self.image_to_show.shape[1]-3]
            cv2.imshow("sni.image_to_show", template)
            cv2.waitKey()
            cv2.destroyAllWindows()
            self.template=template
            
               
            self.w = template.shape[0]
            self.h = template.shape[1]
            # All the 6 methods for comparison in a list

        idx=-1
        for meth in methods:
            self.meth=meth
            #img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv2.matchTemplate(img,self.template,method)
           
            self.res=res
           # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            threshold = 0.50
            loc = np.where( res >= threshold)
            loc_values = res[res >= threshold]
            max_loc=np.argsort(loc_values)[::-1]
            accuracy_value=loc_values[max_loc[0]]
            self.accuracy_value_pre=accuracy_value
            loc_x=loc[1][max_loc]
            loc_y=loc[0][max_loc]
            res_compute=cv2.minMaxLoc(res)
            
            self.compute_accuracy(methods)
            self.dict_acc.update({"Accuracy_pre_"+str(meth):self.accuracy_value_pre})
         #   self.accuracy_value=(self.accuracy_value_pre-self.min_res_value_float)/(self.max_res_value_float-self.min_res_value_float)
            if method==0 or method==1:
                self.acc_minus(method)
            elif method==2 or method==3:
                self.acc_multiplication(method)
            else:
                self.acc_multiplication_1(method)
            
            self.dict_acc.update({"Accuracy_"+str(meth):self.accuracy})
            self.compute_accuracy_0(methods)
      
            
           # number_of_values=input("Write number of max ")
            for idxx in range(1):
                idx+=1
                pt=tuple([loc_x[idx],loc_y[idx]])
                
                if idx==0:
                    matched_center_start=np.asarray([pt[0],pt[1]])
                    matched_center=matched_center_start+np.asarray([np.int64(np.round(self.w/2)),np.int64(np.round(self.h/2))])
                    matched_center=np.resize(matched_center,(1,2))
                    matched_center_array=matched_center
                else:
                    matched_center_start=np.asarray([pt[0],pt[1]])
                    matched_center=matched_center_start+np.asarray([np.int64(np.round(self.w/2)),np.int64(np.round(self.h/2))])
                    matched_center=np.resize(matched_center,(1,2))
                    matched_center_array=np.concatenate((matched_center_array,matched_center),axis=0)
        
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
         #       if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
         #           top_left = min_loc
         #       else:
         #           top_left = max_loc
           #     bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img,pt, (pt[0] + self.h, pt[1] + self.w), (10,240,10), 4)
                cv2.putText(img,"Detected object", (pt[0]+100, pt[1]-10),cv2.FONT_HERSHEY_SIMPLEX,2,(200, 30, 30),3,cv2.LINE_AA)
                cv2.putText(img,("Detetected: ["+str(pt[1])+", "+str(pt[0])+"]"), (pt[0]+100, pt[1]-70),cv2.FONT_HERSHEY_SIMPLEX,2,(220, 50, 50),3,cv2.LINE_AA)
                cv2.putText(img,("Template: ["+str(self.template_s_y+4)+", "+str(self.template_s_x+4)+"]"), (pt[0]+100, pt[1]-130),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 200, 50),3,cv2.LINE_AA)                
                cv2.putText(img,("Accuracy: "+str(np.round(self.accuracy*100,2))+"%"), (pt[0]+100, pt[1]-190),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 200, 50),3,cv2.LINE_AA)
                cv2.putText(img,("Method: "+str(meth)), (pt[0]+100, pt[1]-250),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 50, 250),3,cv2.LINE_AA)
        self.pt=pt
        self.matched_template=img[pt[1]:(pt[1]+self.template.shape[0]),pt[1]:pt[1]+self.template.shape[1]]
        self.moving_image(img)
        self.res_compute=res_compute
        
        large_image=np.concatenate((picture_2[:,:(self.template_s_x+4)],img[:,pt[0]:]),axis=1)
        self.large_image=large_image
      
        sni.moving_image(large_image)
        self.output_image=large_image
        self.large_matrix_x_pre=np.concatenate((self.matrix_x[name_picture_2][:,:(self.template_s_x+4)],self.matrix_x[name_picture_1][:,pt[0]:]),axis=1)
        self.matrix_x.update({(name_picture_2+str(name_picture_1)):self.large_matrix_x_pre})
        self.large_matrix_y_pre=np.concatenate((self.matrix_y[name_picture_2][:,:(self.template_s_x+4)],self.matrix_y[name_picture_1][:,pt[0]:]),axis=1)
        self.matrix_y.update({(name_picture_2+str(name_picture_1)):self.large_matrix_y_pre})
        self.large_matrix_z_pre=np.concatenate((self.matrix_z[name_picture_2][:,:(self.template_s_x+4)],self.matrix_z[name_picture_1][:,pt[0]:]),axis=1)
        self.matrix_z.update({(name_picture_2+str(name_picture_1)):self.large_matrix_z_pre})
    #    self.matched_matrix()
        
    def compute_accuracy(self,methods):        
        template_oposite=np.zeros((self.template.shape),dtype="uint8")
        template_oposite_coor=np.where(self.template<128)
        template_oposite[template_oposite_coor[0],template_oposite_coor[1],template_oposite_coor[2]]=255
        
        for meth in methods:
            #img = img2.copy()
            method = eval(meth)
            print(method)
            max_res_value = cv2.matchTemplate(self.template,self.template,method)
            self.max_res_value_float=np.float64(max_res_value)

        
        for meth in methods:
            #img = img2.copy()
            method = eval(meth)
            min_res_value = cv2.matchTemplate(template_oposite,self.template,method)
            self.min_res_value_float=np.float64(min_res_value)
        
    def acc_minus(self,method):
        zero=np.zeros((self.template.shape),dtype="uint8")
        white=np.full((self.template.shape),255,dtype="uint8")
        ZW_res_value = cv2.matchTemplate(zero,white,method)
        self.ZW=np.float64(ZW_res_value)       
        self.accuracy=(self.ZW-self.accuracy_value_pre)/self.ZW

    def acc_multiplication(self,method):
        white=np.full((self.template.shape),255,dtype="uint8")
        TT_res_value = cv2.matchTemplate(self.template,self.template,method)
        self.TT=np.float64(TT_res_value)
        WW_res_value = cv2.matchTemplate(white,white,method)
        self.WW=np.float64(WW_res_value)
        self.accuracy=(self.WW-np.abs(self.TT-self.accuracy_value_pre))/self.WW
        
    def acc_multiplication_1(self,method):
        TT_res_value = cv2.matchTemplate(self.template,self.template,method)
        self.TT=np.float64(TT_res_value)
        self.accuracy=(self.TT-np.abs(self.TT-self.accuracy_value_pre))/self.TT
        

            
    def compute_accuracy_0(self,methods):        
        template_zero=np.zeros((self.template.shape),dtype="uint8")
        template_white=np.full(self.template.shape, 255,dtype="uint8")

        for meth in methods:
            #img = img2.copy()
            method = eval(meth)
            print(method)
            max_res_value = cv2.matchTemplate(template_zero,template_white,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"zero_white_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(template_zero,template_zero,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"zero_zero_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(template_white,template_white,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"white_white_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(template_white,self.template,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"white_template_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(template_zero,self.template,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"zero_template_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(self.template, template_white,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"template_white_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(self.template,template_zero,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"template_zero_"+str(meth):self.max_res_value_0})
            
            max_res_value = cv2.matchTemplate(self.template,self.template,method)
            self.max_res_value_0=np.float64(max_res_value)
            self.dict_acc.update({"template_template_"+str(meth):self.max_res_value_0})
            
    
    def matched_matrix(self):     
        if hasattr(self,"large_matrix_x"):
            self.large_matrix_x=np.concatenate((self.large_matrix_x,self.large_matrix_x_pre),axis=1)
        else:
            self.large_matrix_x= self.large_matrix_x_pre
        if hasattr(self,"large_matrix_y"):
            self.large_matrix_y=np.concatenate((self.large_matrix_y,self.large_matrix_y_pre),axis=1)   
        else:
            self.large_matrix_y= self.large_matrix_y_pre
        if hasattr(self,"large_matrix_z"):
            self.large_matrix_z=np.concatenate((self.large_matrix_z,self.large_matrix_z_pre),axis=1)   
        else:
            self.large_matrix_z= self.large_matrix_z_pre

        
        
    def cut_picture(self,picture_1,methods,name_picture_1):
        img=np.copy(picture_1)
        self.init_image(img)
        self.snipping()
        template_s_x=self.s_x_m+self.udx*1000
        template_s_y=self.s_y_m
        
        template = self.image_to_show[3:self.image_to_show.shape[0]-3,3:self.image_to_show.shape[1]-3]
        cv2.imshow("sni.image_to_show", template)
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.template=template
        
           
        w = template.shape[0]
        h = template.shape[1]
        # All the 6 methods for comparison in a list

        idx=-1
        for meth in methods:
            #img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            
            self.res=res
           # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            threshold = 0.50
            loc = np.where( res >= threshold)
            loc_values = res[res >= threshold]
            max_loc=np.argsort(loc_values)[::-1]
            max_loc_values=np.sort(loc_values)
            loc_x=loc[1][max_loc]
            loc_y=loc[0][max_loc]
            res_compute=cv2.minMaxLoc(res)
            
           # number_of_values=input("Write number of max ")
            for idxx in range(3):
                idx+=1
                pt="pt"+str(idxx)
                vars()[pt]=tuple([loc_x[idx],loc_y[idx]])
                
                if idx==0:
                    matched_center_start=np.asarray([vars()[pt][0],vars()[pt][1]])
                    matched_center=matched_center_start+np.asarray([np.int64(np.round(w/2)),np.int64(np.round(h/2))])
                    matched_center=np.resize(matched_center,(1,2))
                    matched_center_array=matched_center
                else:
                    matched_center_start=np.asarray([vars()[pt][0],vars()[pt][1]])
                    matched_center=matched_center_start+np.asarray([np.int64(np.round(w/2)),np.int64(np.round(h/2))])
                    matched_center=np.resize(matched_center,(1,2))
                    matched_center_array=np.concatenate((matched_center_array,matched_center),axis=0)
        
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
         #       if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
         #           top_left = min_loc
         #       else:
         #           top_left = max_loc
           #     bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img,vars()[pt], (vars()[pt][0] + h, vars()[pt][1] + w), (10,240,10), 4)
                cv2.putText(img,"Detected object", (vars()[pt][0], vars()[pt][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
                cv2.putText(img,("Detected: ["+str(vars()[pt][1])+", "+str(vars()[pt][0])+"]"), (vars()[pt][0], vars()[pt][1]-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
                cv2.putText(img,("Template: ["+str(template_s_y+4)+", "+str(template_s_x+4)+"]"), (vars()[pt][0], vars()[pt][1]-80),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 250, 0),2,cv2.LINE_AA) 
              
              #  cv2.putText(img,("Accuracy: "+str(np.round(res_compute[1]*100,2))+"%"), (vars()[pt][0], vars()[pt][1]-120),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 250, 0),2,cv2.LINE_AA)
                self.pt_name=pt
                self.pt_dict.update({self.pt_name : tuple([loc_x[idx],loc_y[idx]])})
        self.cutted_image=img[:,self.pt_dict["pt"+str(0)][0]:self.pt_dict["pt"+str(1)][0],:]
        self.moving_image(img)
        self.res_compute=res_compute
        self.moving_image(self.cutted_image)
        self.matrix_x.update({name_picture_1+"_cutted":self.matrix_x["picture_21picture_11picture_41"][:,self.pt_dict["pt"+str(0)][0]:self.pt_dict["pt"+str(1)][0],:]})
        self.matrix_y.update({name_picture_1+"_cutted":self.matrix_y["picture_21picture_11picture_41"][:,self.pt_dict["pt"+str(0)][0]:self.pt_dict["pt"+str(1)][0],:]})
        self.matrix_z.update({name_picture_1+"_cutted":self.matrix_z["picture_21picture_11picture_41"][:,self.pt_dict["pt"+str(0)][0]:self.pt_dict["pt"+str(1)][0],:]})
        
    def create_dataset(self):
     #   self.dataset_templates=dict()
        GUI_image_label_exist=np.copy(self.GUI["label_exist"])
        #for idx in range(11):
        picture_keys=list(self.dict_pictures.keys())
        while True:
            sample=False
            self.labeling("green","Number of scan: ",picture_keys,sample)
            try:
                idx=int(self.label[:1])
    
                self.name_pic="picture_"+str(idx)+"1"
                name_pic_GUI="picture_"+str(idx)
                self.matrix_x_idx=self.matrix_x[self.name_pic]
                self.matrix_y_idx=self.matrix_y[self.name_pic]
                self.matrix_z_idx=self.matrix_z[self.name_pic]
                self.picture_idx=self.dict_pictures[self.name_pic]
                
    
                
                k=ord("e")
                print("start loop: ")
                idxx=0
                image_loop=np.copy(self.dict_pictures[self.name_pic])
                sample=True
                try:
                    self.dataset_picture=self.dataset_templates[self.name_pic]
                except:
                    self.dataset_picture=dict()
                while k!=ord("q"):
                    idxx+=1
                    self.init_image(image_loop)
                    self.snipping()
                    self.save_template = np.copy(self.image_to_show[3:self.image_to_show.shape[0]-3,3:self.image_to_show.shape[1]-3])
                    self.matrix_point=np.asarray([self.matrix_x_idx[(self.B_S_VDX+self.s_y+3):(self.B_S_VDX+self.e_y-3),(self.B_S_HDX+self.s_x+3):(self.B_S_HDX+self.e_x-3)],
                                                  self.matrix_x_idx[(self.B_S_VDX+self.s_y+3):(self.B_S_VDX+self.e_y-3),(self.B_S_HDX+self.s_x+3):(self.B_S_HDX+self.e_x-3)],
                                                  self.matrix_x_idx[(self.B_S_VDX+self.s_y+3):(self.B_S_VDX+self.e_y-3),(self.B_S_HDX+self.s_x+3):(self.B_S_HDX+self.e_x-3)]])                   
                    
                    GUI_image_to_save=np.copy(self.GUI["saving_sample"])
                    
                    cv2.imshow("GUI",GUI_image_to_save)
                    button=cv2.waitKey(0)
                
                    data_picture_idx_keys= list(self.dataset_picture.keys())
                    if button==ord("y"):
                        cv2.destroyAllWindows()
                        self.labeling("gray","Write name of label: ",data_picture_idx_keys,sample)
                        while self.label in self.dataset_picture:
                            cv2.imshow("Label exist!",GUI_image_label_exist)
                            cv2.waitKey()
                            cv2.destroyAllWindows()
                            self.labeling("gray","Write name of label: ",data_picture_idx_keys,sample)
                        else:
                            self.dataset_picture.update({self.label:{"Snipped picture":self.save_template,
                                                                     "Picture":self.picture_idx[(self.B_S_VDX+self.s_y+3):(self.B_S_VDX+self.e_y-3),(self.B_S_HDX+self.s_x+3):(self.B_S_HDX+self.e_x-3)],
                                                                     "Point cloud":self.matrix_point,
                                                                     "Positions": (str(self.B_S_VDX+self.s_y+3)+":"+str(self.B_S_VDX+self.e_y-3)+","+str(self.B_S_HDX+self.s_x+3)+":"+str(self.B_S_HDX+self.e_x-3)),
                                                                     "Coordinates":np.asarray([(self.B_S_VDX+self.s_y+3),(self.B_S_VDX+self.e_y-3),(self.B_S_HDX+self.s_x+3),(self.B_S_HDX+self.e_x-3)])}}) 
                            
                           
                    else:
                        cv2.destroyAllWindows()
                    
                    GUI_image_to_exit=np.copy(self.GUI["exit_loop"]) 
                    cv2.putText(GUI_image_to_exit,(name_pic_GUI), (390,115),cv2.FONT_HERSHEY_SIMPLEX,1,(220, 220, 220),2,cv2.LINE_AA)
                    cv2.imshow("GUI",GUI_image_to_exit)
                    k=cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(chr(k)+": was pressed")
                    cv2.rectangle(image_loop,(self.B_S_HDX+self.s_x,self.B_S_VDX+self.s_y), (self.B_S_HDX+self.e_x, self.B_S_VDX+self.e_y), (10,240,10), 4)
                    
                self.dataset_templates.update({self.name_pic:self.dataset_picture})
                    
                
            except:
                pass
                    
            if self.label=="q":
                
                break
        
    def labeling(self, name_of_picture,text,list_item,sample):
        GUI_pic=np.copy(self.GUI[name_of_picture])
        GUI_pic_shape=GUI_pic.shape
        
        k="D"
        font = cv2.FONT_HERSHEY_SIMPLEX
        label=str()
        i=0
        cv2.putText(GUI_pic,text,(10,50), font, 2,(0,0,0),3,cv2.LINE_AA)
        if sample==True:
            self.im_resize(self.image_to_save,"sample", (200,150),"N")
            GUI_pic[10:self.shape_resize[1]+10,(GUI_pic_shape[1]-(self.shape_resize[0]+400)):(GUI_pic_shape[1]-400),:]=self.image_resize
        
        while True:
            cv2.imshow("label",GUI_pic)
            k=cv2.waitKey()
            if k==8:
                i-=1
                label=label[:-1]
                
            else:
                i+=1
                label=label+chr(k)
            matching = [s for s in list_item if label[:2] in s]
            GUI_pic=np.copy(sni.GUI[name_of_picture])
            if sample==True:
                GUI_pic[10:self.shape_resize[1]+10,(GUI_pic_shape[1]-(self.shape_resize[0]+400)):(GUI_pic_shape[1]-400),:]=self.image_resize
            cv2.putText(GUI_pic,text,(10,50), font, 2,(0,0,0),3,cv2.LINE_AA)
            for posi,word in enumerate(matching):
                word=str(word[:-1])
                cv2.putText(GUI_pic,word,(GUI_pic_shape[1]-350,120+posi*70), font, 2,(0,0,0),3,cv2.LINE_AA)
                
           
            cv2.putText(GUI_pic,label,(20,120), font, 2,(255,255,255),3,cv2.LINE_AA)
            
            
            if k==ord("q") or k==27 or k==13:
                break
        cv2.destroyAllWindows()
        self.label=label
    
    def im_resize(self,image,name,shape, save):
        self.shape_resize=shape
        self.image_resize=cv2.resize(image,(shape))
        if save=="Y":
            cv2.imwrite(self.path+name+".png",self.image_resize)
        

    def moving(self):
        k = self.k
        shape=self.image_to_show_big.shape
        if k==ord("p"):
            cv2.putText(self.image_to_show,("["+str(self.s_y)+", "+str(self.s_x+self.udx*1000)+"]"), (self.s_x, self.s_y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0, 250, 0),2,cv2.LINE_AA)


        if (self.udx>=0 or self.udx<int(shape[1]/300)):
            if (k==ord("d")) and (self.udx<int((shape[1]/300)-2)):
                self.udx+=1
            if k==ord("a") and (self.udx>=1):
               self.udx-=1 
        if (self.vdx>=0 or self.vdx<int(shape[0]/100)):  
      
            if k==ord("w") and (self.vdx>=1):
                self.vdx-=1 
            if k==ord("s") and self.vdx<(int(shape[0]/100)-2):
                self.vdx+=1
            
        self.B_S_VDX=self.vdx*100
        self.B_S_HDX=self.udx*300
        
        if 800>shape[0]:
            self.B_E_VDX=shape[0]
            self.B_S_VDX=0

        elif ((self.vdx+1)*100+700)>shape[0]:
            self.B_E_VDX=shape[0]
            self.B_S_VDX=self.B_E_VDX-800

        else:
            self.B_E_VDX=((self.vdx+1)*100+700)
            
        if 1900>shape[1]:   
            self.B_E_HDX=shape[1]
            self.B_S_HDX=0
        elif ((self.udx+1)*300+1600)>shape[1]:
            self.B_E_HDX=shape[1]
            self.B_S_HDX=self.B_E_HDX-1900
        else:
            self.B_E_HDX=((self.udx+1)*300+1600)
            
    def finding_samples(self,dict_samples,picture,methods):
        img=np.copy(picture)
       
        idx=-1
        for meth in methods:
            self.meth=meth
            #img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv2.matchTemplate(img,self.template,method)
           
            self.res=res
           # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            threshold = 0.50
            loc = np.where( res >= threshold)
            loc_values = res[res >= threshold]
            max_loc=np.argsort(loc_values)[::-1]
            accuracy_value=loc_values[max_loc[0]]
            self.accuracy_value_pre=accuracy_value
            loc_x=loc[1][max_loc]
            loc_y=loc[0][max_loc]
            res_compute=cv2.minMaxLoc(res)
            
            self.compute_accuracy(methods)
            self.dict_acc.update({"Accuracy_pre_"+str(meth):self.accuracy_value_pre})
         #   self.accuracy_value=(self.accuracy_value_pre-self.min_res_value_float)/(self.max_res_value_float-self.min_res_value_float)
            if method==0 or method==1:
                self.acc_minus(method)
            elif method==2 or method==3:
                self.acc_multiplication(method)
            else:
                self.acc_multiplication_1(method)
            
            self.dict_acc.update({"Accuracy_"+str(meth):self.accuracy})
            self.compute_accuracy_0(methods)
      
            
           # number_of_values=input("Write number of max ")
            for idxx in range(1):
                idx+=1
                pt=tuple([loc_x[idx],loc_y[idx]])
                
                if idx==0:
                    matched_center_start=np.asarray([pt[0],pt[1]])
                    matched_center=matched_center_start+np.asarray([np.int64(np.round(self.w/2)),np.int64(np.round(self.h/2))])
                    matched_center=np.resize(matched_center,(1,2))
                    matched_center_array=matched_center
                else:
                    matched_center_start=np.asarray([pt[0],pt[1]])
                    matched_center=matched_center_start+np.asarray([np.int64(np.round(self.w/2)),np.int64(np.round(self.h/2))])
                    matched_center=np.resize(matched_center,(1,2))
                    matched_center_array=np.concatenate((matched_center_array,matched_center),axis=0)
        
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
         #       if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
         #           top_left = min_loc
         #       else:
         #           top_left = max_loc
           #     bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img,pt, (pt[0] + self.h, pt[1] + self.w), (10,240,10), 4)
                cv2.putText(img,"Detected object", (pt[0]+100, pt[1]-10),cv2.FONT_HERSHEY_SIMPLEX,2,(200, 30, 30),3,cv2.LINE_AA)
                cv2.putText(img,("Detetected: ["+str(pt[1])+", "+str(pt[0])+"]"), (pt[0]+100, pt[1]-70),cv2.FONT_HERSHEY_SIMPLEX,2,(220, 50, 50),3,cv2.LINE_AA)
                cv2.putText(img,("Template: ["+str(self.template_s_y+4)+", "+str(self.template_s_x+4)+"]"), (pt[0]+100, pt[1]-130),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 200, 50),3,cv2.LINE_AA)                
                cv2.putText(img,("Accuracy: "+str(np.round(self.accuracy*100,2))+"%"), (pt[0]+100, pt[1]-190),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 200, 50),3,cv2.LINE_AA)
                cv2.putText(img,("Method: "+str(meth)), (pt[0]+100, pt[1]-250),cv2.FONT_HERSHEY_SIMPLEX,2,(50, 50, 250),3,cv2.LINE_AA)
        self.pt=pt
        self.matched_template=img[pt[1]:(pt[1]+self.template.shape[0]),pt[1]:pt[1]+self.template.shape[1]]
        self.moving_image(img)
        self.res_compute=res_compute
        
    def samples(self, dict_samples):
        
        if type(dict_samples)==dict:
            
            
            dict_samples_keys=list(dict_samples.keys())
            self.idx=(np.arange(len(dict_samples_keys)))
            k="r"
            self.image_show=np.zeros((900,1950,3),dtype="uint8") 
            self.idxx=np.arange(2)
            for idy in range(3): 
                dict_pic_samples=dict_samples[dict_samples_keys[self.idx[idy]]]
                dict_pic_samples_keys=list(dict_pic_samples.keys())
                if len(self.idxx)<len(dict_pic_samples_keys):
                    self.idxx=(np.arange(len(dict_pic_samples_keys)))
                
            while True:
                coor_y=int(self.image_show.shape[0]/3)
                  
                self.image_show=np.zeros((900,1950,3),dtype="uint8")  
              
                if k==ord("w"):
                    self.idx=np.insert(self.idx[1:],(len(self.idx)-1),self.idx[0])
                if k==ord("s"):
                    self.idx=np.insert(self.idx[:-1],0,self.idx[-1])

                if k==ord("d"):
                    self.idxx=np.insert(self.idxx[1:],(len(self.idxx)-1),self.idxx[0])
                    
                if k==ord("a"):
                    self.idxx=np.insert(self.idxx[:-1],0,self.idxx[-1])

                for idy in range(3):                
                    cv2.putText(self.image_show,(dict_samples_keys[self.idx[idy]]),(10,160+(idy)* coor_y),cv2.FONT_HERSHEY_SIMPLEX,1,(150,150,150),2,cv2.LINE_AA )
                    dict_pic_samples=dict_samples[dict_samples_keys[self.idx[idy]]]
                    dict_pic_samples_keys=list(dict_pic_samples.keys())

                    try:
                        for idyy in range(3):
                            dict_picture=dict_pic_samples[dict_pic_samples_keys[self.idxx[idyy]]]
                            picture=dict_picture["Snipped picture"]
                            self.put_image(picture,idy,idyy)
                            if idy==1 and idyy==1:
                                self.output_image=dict_picture
                                
                    except:
                        pass
                cv2.imshow("Samples",self.image_show)
                k=cv2.waitKey()
                if k==ord("q") or k==27 or k==13:
                    break
                
            cv2.destroyAllWindows()
        
    def put_image(self,image,posi_y,posi_x):
        
        self.image_test=image
        image_shape=image.shape
        ratio=image_shape[1]/image_shape[0]
        default_ratio=550/280
        if ratio<default_ratio:
            new_ratio=image_shape[0]/280
            new_size=(int(image_shape[1]/new_ratio),int(image_shape[0]/new_ratio))

        else:
            new_ratio=image_shape[1]/550
            new_size=(int(image_shape[1]/new_ratio),int(image_shape[0]/new_ratio))

            
        self.new_image=cv2.resize(image,(new_size))
        self.nim=self.new_image.shape
        self.y_start=int((280-self.nim[0])/2+posi_y*(310))
        self.x_start=int((450-self.nim[1]/2)+posi_x*(565))
        self.y_end=self.y_start+self.nim[0]
        self.x_end=self.x_start+self.nim[1]
        self.image_test_new=self.new_image
        self.image_put=self.image_show[self.y_start:self.y_end,self.x_start:self.x_end,:]
        if self.image_put.shape==self.new_image.shape:
            self.image_show[self.y_start:self.y_end,self.x_start:self.x_end,:]=self.new_image[:,:,:] 
            
    def find_sample(self,picture,sample_dict,methods,name_picture,label_image):
        img=np.copy(picture)
        self.label_image=label_image
        sample_dict_keys=sample_dict.keys()
        for label in sample_dict_keys:
            smaple_dict=sample_dict[label]
            sample=smaple_dict["Snipped picture"]
            idx=-1
            for meth in methods:
                self.meth=meth
                #img = img2.copy()
                method = eval(meth)
                # Apply template Matching
                res = cv2.matchTemplate(img,sample,method)
               
                self.res=res
               # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                threshold = 0.50
                loc = np.where( res >= threshold)
                loc_values = res[res >= threshold]
                max_loc=np.argsort(loc_values)[::-1]
                accuracy_value=loc_values[max_loc[0]]
                self.accuracy_value_pre=accuracy_value
                loc_x=loc[1][max_loc]
                loc_y=loc[0][max_loc]
                res_compute=cv2.minMaxLoc(res)
          
                
               # number_of_values=input("Write number of max ")
                for idxx in range(1):
                    idx+=1
                    pt=tuple([loc_x[idx],loc_y[idx]])
                    
                    if idx==0:
                        matched_center_start=np.asarray([pt[0],pt[1]])
                        matched_center=matched_center_start+np.asarray([np.int64(np.round(sample.shape[1]/2)),np.int64(np.round(sample.shape[0]/2))])
                        matched_center=np.resize(matched_center,(1,2))
                        matched_center_array=matched_center
                    else:
                        matched_center_start=np.asarray([pt[0],pt[1]])
                        matched_center=matched_center_start+np.asarray([np.int64(np.round(sample.shape[1]/2)),np.int64(np.round(sample.shape[0]/2))])
                        matched_center=np.resize(matched_center,(1,2))
                        matched_center_array=np.concatenate((matched_center_array,matched_center),axis=0)
            
                    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
             #       if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
             #           top_left = min_loc
             #       else:
             #           top_left = max_loc
               #     bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(img,pt, (pt[0] + sample.shape[1], pt[1] + sample.shape[0]), (10,240,10), 4)
                    label=str(label)
                    if label.find("_"):
                        index=label.find("_")
                        label_str=label[:index]
                    else:
                        label_str=label[:-1]
                
                    print(label) 
                    print(label_str) 
                    
                    cv2.putText(img,label_str, (pt[0]+5, pt[1]-10),cv2.FONT_HERSHEY_SIMPLEX,2,(200, 30, 30),3,cv2.LINE_AA)
                self.images_coordinates.update({name_picture:{name_picture:pt,"Sample shape":sample.shape,"Image":img}})
        
        self.init_image(img)
        self.moving_image(img)
        self.res_compute=res_compute

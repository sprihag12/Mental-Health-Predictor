import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import seaborn as sns
from scipy import stats
import warnings
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import csv
#get the library to handle multipoint
from shapely.geometry import MultiPoint
import os

data1 = pd.read_csv("survey.csv")
new_df=data1
new_df = new_df.replace(["Somewhat easy","Somewhat difficult", "Very difficult", "Very easy", "M", "Male", "male", "m", "Female","female","f","F","Male-ish","maile","Cis Male","Cis Female", "Woman", "Female ", "Make"], [1,2,3,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0])

new_df = new_df.replace(["Yes","No", "Maybe", "Don't know", "Not sure", "Never","Rarely", "Sometimes","Often", "Usually", "Some of them"], [1,0,2,0,0,0,1,2,3,3,2])
new_df.fillna(3, inplace=True)

y = new_df.loc[:,'treatment'].values

columns = ['Age', 'Gender','self_employed','family_history','work_interfere','care_options','mental_health_consequence','wellness_program','mental_health_interview','obs_consequence']
columns1 = ['Age','Gender','family_history','obs_consequence']

X = new_df.loc[:,columns].values
X1 = new_df.loc[:,columns1].values

logit = sm.Logit(y, X1)
result = logit.fit()

print('Enter age')
age = input()
print('Enter gender, 0 for male and 1 for female')
gender = input()
print('Do you have a family history of mental illnesses? Type 0 for No, Type 1 for Yes')
famIllness = input()
print('Do you live in a stressful environment? Type 0 for No, Type 1 for Yes')
healthCons = input()
outvalue = result.params[0]*float(age) +  result.params[1]*float(gender) + result.params[2]*float(famIllness) + result.params[3]*float(healthCons)
resultOfIllness = (1/(1+math.exp(-outvalue))) #Logistic Regression result


def option():
    print(" 1.Always")
    print(" 2.Usually")
    print(" 3.Sometimes")
    print(" 4.Rarely")
    print(" 5.Never")

def illness_level():
    #Ask survey questions related to depression
    print("I feel sad")
    option()
    dep_q1=float(input("Enter one of the above options:"));
    print("I feel worn out")
    option()
    dep_q2=float(input("Enter one of the above options:"));
    print("I feel so guilty that I can barely take it")
    option()
    dep_q3=float(input("Enter one of the above options:"));
    print("When things go wrong in my life, I feel like I will never get over it")
    option()
    dep_q4=float(input("Enter one of the above options:"));
    print("When I wake up in the morning, I feel like there is nothing to look forward to.")
    option()
    dep_q5=float(input("Enter one of the above options:"));
    final_dep_score=dep_q1*0.15+dep_q2*0.15+dep_q3*0.25+dep_q4*0.25+dep_q5*0.2
    
    print("I am not able to relax")
    option()
    anx_q1=float(input("Enter one of the above options:"));
    print("I feel fearful for no reason")
    option()
    anx_q2=float(input("Enter one of the above options:"));
    print("When someone snaps at me, I spend the rest of the day thinking about it")
    option()
    anx_q3=float(input("Enter one of the above options:"));
    print("I am easily alarmed, frightened, or surprised")
    option()
    anx_q4=float(input("Enter one of the above options:"));
    print("I am afraid of crowds / being left alone / the dark / of strangers / of traffic")
    option()
    anx_q5=float(input("Enter one of the above options:"));
    print("I have trouble falling or staying asleep")
    option()
    anx_q6=float(input("Enter one of the above options:"));
    final_anx_score=anx_q1*0.15+anx_q2*0.15+anx_q3*0.20+anx_q4*0.15+anx_q5*0.20+anx_q6*0.15
    #print(final_anx_score)
    print("I have recurrent distressing dreams or memories of a life-threatening event that I experienced in the past")
    option()
    ptsd_q1=float(input("Enter one of the above options:"));
    print("I have trouble concentrating or recalling events")
    option()
    ptsd_q2=float(input("Enter one of the above options:"));
    print("I avoid certain types of places that remind me of my past")
    option()
    ptsd_q3=float(input("Enter one of the above options:"));
    print("I have a feeling as if a traumatic event from the past were actually happening all over again i.e. flashbacks")
    option()
    ptsd_q4=float(input("Enter one of the above options:"));
    print("It is difficult for me to imagine my future, such as career, marriage, children, or a normal lifespan")
    option()
    ptsd_q5=float(input("Enter one of the above options:"));
    final_ptsd_score=ptsd_q1*0.25+ptsd_q2*0.15+ptsd_q3*0.15+ptsd_q4*0.25+ptsd_q5*0.2
    #print(final_ptsd_score)
    final_score=min(final_dep_score,final_anx_score,final_ptsd_score)
    if final_score<3:
        level=3
    else:
        level=2
    if final_score==final_ptsd_score:
        return level,"PTSD"
    elif final_score==final_anx_score:
        return level,"Anxiety"
    else:
        return level,"Depression"
        
if(resultOfIllness > 0.6):
    level,illness =illness_level()
else:
    print("You Are Fine!")


#function to make recommendations to the user based on the predicted mental health index and category
def find_help(score,category):
    
    #a score of 1 indicates safe zone: no mental illness predicted
    if score ==1:
        print("Awesome! You are good to go! Just a bunch of activities you can try when you feel stressed out:")
        df = pd.read_csv("RecreationalActivities.csv")
        for index,row in df.iterrows():
            print(colored(row[0].upper(),'green')+':'+'\n'+row[1]+'\n\n')
     
    #a score of 2 indicates a moderate mental health condition 
    elif score==2:
        print("You are likely to have Moderate "+category)
        print("But don't worry,you are not alone! There are support groups to help you cope with this:\n")
        
        #read a file containing support group data based on the category
        if(category.lower()=='depression'):
            df = pd.read_csv("SupportGroups_Depression.csv")
            for index,row in df.iterrows():
                #extracting the relevant fields like supportgroup name, address, contact number from the csv file

                print(row[0]+'\n'+row[2]+'\n'+'Office: '+str(row[6])+','+str(row[7])+','+str(row[8]))
                print('You can call them at: '+row[3]+'\n')
                
        elif category.lower() =='anxiety':
            df = pd.read_csv("SupportGroups_Anxiety.csv")
            for index,row in df.iterrows():
                print(row[0]+'\n'+row[2]+'\n'+'Office: '+str(row[5])+','+str(row[6])+','+str(row[7]))
                print('You can call them at: '+row[3]+'\n')
                
        else:
            df = pd.read_csv("SupportGroups_PTSD.csv")
            for index,row in df.iterrows():
                print(row[0]+'\n'+row[2]+'\n'+'Office: '+str(row[5])+','+str(row[6])+','+str(row[7]))
                print('You can call them at: '+row[3]+'\n')
                
    
    #a score of 3 indicates that the person needs professional help           
    else:
        print("You seem to show high levels of "+category)
        print("But don't worry, you are not alone! There are experienced professionals to help you out:\n")

        #Using csv reader to read the Therapists data
        with open("Therapists.csv") as fin:
            csv_reader = csv.reader(fin,delimiter =',')
            count = 0
            for row in csv_reader:
                count+=1
                if(count==1 ):
                    continue
                else:
                    print(row[0]+'\n'+'Office: '+row[1]+'\n'+'To make an appointment call: '+row[2].strip()+'\n'+'Address: '+row[4]+'\n')
            
if __name__ =='__main__' and resultOfIllness>0.6:
    find_help(level,illness)
    
want_data=input("You are not alone! Do you want to know more about mental health? y/n :")
if want_data=='y':
    # getting the data for the Top 10 States with highest Male percentage with mental illness
    data = pd.read_csv("states_percentage.csv")

    #Getting the cordinates for the 10 states into a list of tuples
    points = [(-120.5380993,44.1419049),(-111.547028,39.4997605),(-92.1313784,34.7519275),(-71.718067,42.0629398),
         (-86.441277,39.7662195),(-91.4299097,30.9733766),(-105.550567,38.9979339),(-84.4158049,44.9435598),
         (-77.6046984,40.9945928),(-120.0145665,47.8993487)]
    #Location Points for formatting the State 
    point1 = [(-120.5380993,44.1419049)]
    point2 = [(-111.547028,39.4997605)]
    point3 = [(-92.1313784,34.7519275)]
    point4 = [(-71.718067,42.0629398)]
    point5 = [(-86.441277,39.7662195)]
    point6 = [(-91.4299097,30.9733766)]
    point7 = [(-105.550567,38.9979339)]
    point8 = [(-84.4158049,44.9435598)]
    point9 = [(-77.6046984,40.9945928)]
    point10 = [(-120.0145665,47.8993487)]

    #Setting the Geographical map of the US state
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)

    # variable ax contains the figure

    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75], projection=ccrs.LambertConformal())

    plt.autoscale(enable=True, axis='both', tight=None)
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())


    #Getting the US states outline shapes from the Cartopy reader Library
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='110m',category='cultural', name=shapename)
    ax.background_patch.set_visible(True)
    ax.outline_patch.set_visible(True)

    # setting the Title for the Graph
    ax.set_title('Top U.S. states with the highest percentage of poor mental health among adults 2017')


    # Creating a track of all the points relating to the states
    #to display against the map
    track = sgeom.MultiPoint(points)

    # creating Track for each of the states' Capitals' locations
    track1 =  sgeom.Point(point1)
    track2 =  sgeom.Point(point2)
    track3 =  sgeom.Point(point3)
    track4 =  sgeom.Point(point4)
    track5 =  sgeom.Point(point5)
    track6 =  sgeom.Point(point6)
    track7 =  sgeom.Point(point7)
    track8 =  sgeom.Point(point8)
    track9 =  sgeom.Point(point9)
    track10 = sgeom.Point(point10)

    # defining a styler function to get seperate color for each state
    # if a the graph shape coinsides with a location then the state color changed
    def colorize_state(geometry):
            facecolor = (0.9375, 0.9375, 0.859375)
            if geometry.intersects(track1):
                facecolor = 'yellow'
            if geometry.intersects(track2):
                facecolor = 'orange'
            if geometry.intersects(track3):
                facecolor = 'green'
            if geometry.intersects(track4):
                facecolor = 'blue'
            if geometry.intersects(track5):
                facecolor = 'pink'
            if geometry.intersects(track6):
                facecolor = 'violet'
            if geometry.intersects(track7):
                facecolor = 'cyan'
            if geometry.intersects(track8):
                facecolor = 'lawngreen'
            if geometry.intersects(track9):
                facecolor = 'red'
            if geometry.intersects(track10):
                facecolor = 'skyblue'
            return {'facecolor': facecolor, 'edgecolor': 'black'}

    # to the ax, add the US map geometries, and the styler  
    ax.add_geometries(shpreader.Reader(states_shp).geometries(),ccrs.PlateCarree(),styler=colorize_state)

    # giving colors to the states
    Oregon = mpatches.Rectangle((0, 0), 1, 1, facecolor="yellow")
    Utah = mpatches.Rectangle((0, 0), 1, 1, facecolor="orange")
    Arkansas = mpatches.Rectangle((0, 0), 1, 1, facecolor="green")
    Massachusetts = mpatches.Rectangle((0, 0), 1, 1, facecolor="blue") 
    Indiana = mpatches.Rectangle((0, 0), 1, 1, facecolor="pink")
    Louisiana = mpatches.Rectangle((0, 0), 1, 1, facecolor="violet")
    Colorado = mpatches.Rectangle((0, 0), 1, 1, facecolor="cyan")
    Michigan = mpatches.Rectangle((0, 0), 1, 1, facecolor="lawngreen")
    Pennsylvania = mpatches.Rectangle((0, 0), 1, 1, facecolor="red")
    Washington = mpatches.Rectangle((0, 0), 1, 1, facecolor="skyblue")

    # Generating Legends with the values for each state 
    label = []
    for index,row in data.iterrows() :
        s = row["State_name"]+ ": " + str(row['Percentage'])+str("%")
        label.append(s)

    labels = [label[0],label[1],label[2],label[3],label[4],label[5],label[6],label[7],label[8],label[9]]

    # displaying the legend
    plt.legend([Oregon,Utah,Arkansas,Massachusetts,Indiana,Louisiana,Colorado,Michigan,Pennsylvania,Washington,], 
               labels,loc='lower left', bbox_to_anchor=(1, 0.3),  fancybox=True)

    plt.show()
    

    # get the data into a data frame
    df = pd.read_csv('survey.csv')

    # filtering the data to get data for Men
    filterM = df["Gender"] == 'M'
    df1 = df[filterM]
    filterM_Yes = df1["treatment"]== 'Yes'
    filterM_No = df1["treatment"]== 'No'

    # get data for Men for the treatment = yes
    dfM_Y = df1[filterM_Yes]


    # filtering the data to get data for Female
    filterF = df["Gender"] == 'F'
    df1 = df[filterF]
    filterF_Yes = df1["treatment"]== 'Yes'
    filterF_No = df1["treatment"]== 'No'

    # get data for Female for the treatment = yes
    dfF_Y = df1[filterF_Yes]
    dfF_N = df1[filterF_No]

    # Setting the style for the plot
    sns.set(style="white", palette="muted", color_codes=True)

    # Set the subplots
    f, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    sns.despine(left=True)

    # plotting the plots for the two distributions
    def ploting():
        # Plot
        sns.distplot(tuple(dfM_Y["Age"]),kde=False, rug=True, color="b", ax=axes[0]).set_title("Distribution Plot for Mental Illnes in Men")
        # Plot
        sns.distplot(tuple(dfF_Y["Age"]), kde=False, rug=True,  color="g",  ax=axes[1]).set_title("Distribution Plot for Mental illnes in Women")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ploting()

    # set other attributes and show the plot
    plt.setp(axes, yticks=[])
    plt.tight_layout()

    plt.show()
    
   


    # read the survey data and store in a data frame
    data = pd.ExcelFile("statistic_id727631_sources-of-stress-on-personal-activities-and-or-relationships-us-2017.xlsx")
    df = pd.read_excel(data, sheet_name="Data")

    # Sort the data in ascending order of share of respondents
    sorted_by_share = df.sort_values(['Share of respondents'], ascending=True)

    # Creating the horizontal bar chart with color code m
    ax = sorted_by_share.plot(kind="barh",color="m")
    #Storing every bar of the horizontal bar chart 
    rects = ax.patches
    #For every horizontal bar,display the data values in %
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y()
        label = "{:.0%}".format(x_value/100)
        plt.annotate( label,(x_value, y_value)) 
    #Display title of chart
    plt.title('Sources of stress on personal activities in 2017 amongst youth')    
    plt.show()

    # read the survey data and store in a data frame
    data = pd.ExcelFile("statistic_id796063_us-adults-who-saw-a-health-professional-for-depression-2016-2017-by-type.xlsx")
    df = pd.read_excel(data, sheet_name="Data")
    # Sort the data in ascending order of 2016 year
    sorted_by_share = df.sort_values(['2016'], ascending=True)
    ax = sorted_by_share.plot(kind="barh",color=["orange","lightseagreen"])
    rects = ax.patches
    #For every horizontal bar,display the data values in % till 2 decimal places
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y()
        label = "{:.2f}".format(x_value)
        plt.annotate( label,(x_value, y_value))

    # plot the graph with  x axis and y axis labels and title of the chart
    plt.yticks(rotation=32)
    plt.xlabel('Number of people in millions')
    plt.title('Number of U.S. adults with a major depressive episode who saw a health professional about depression in 2016 and 2017, by type (in millions)')      
    plt.show()


    # read the survey data and store in a data frame
    data = pd.ExcelFile("statistic_id252325_treatment-received-by-us-youths-with-major-depressive-episode-by-gender-2017.xlsx")
    df = pd.read_excel(data, sheet_name="Data")
    sorted_by_share = df.sort_values(['Female'], ascending=True)
    ax = sorted_by_share.plot(kind="barh",color=["gold","mediumblue"])
    rects = ax.patches

    # Creating the plot
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y()
        label = "{:0.2%}".format(x_value/100)
        plt.annotate( label,(x_value, y_value)) 
    plt.xlabel('Percentage of respondents')
    plt.title('Type of treatment received by U.S. youths with a major depressive episode in the past year in 2017, by gender')      
    plt.show()


    

    # Family background history of mental illness in males

    # read the survey data
    df = pd.read_csv('survey.csv')
    #Filter gender with male
    filterM = df["Gender"] == 'M'
    df1 = df[filterM]
    background_data = df1['family_history']
    Category=background_data.unique()
    count_yes=0
    count_no=0
    i=0

    # getting the counts for each category
    for index, row in df1.iterrows(): 
        if  row["family_history"]==Category[0]:
            count_yes+=1  
        if row["family_history"]==Category[1]:
            count_no+=1
    labels = Category[0], Category[1]
    sizes = [count_no,count_yes] #list of counts for existing family history 
    colors = ['darkblue', 'dodgerblue'] #define colours
    explode = (0.1, 0)  # explode 1st slice

    # 
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Family background history of mental illness in males")
    plt.axis('equal')
    plt.show()


    # read the data, Data got from Statista
    df = pd.read_csv('survey.csv')
    #Filtering males with background in family history
    filterM = df["Gender"] == 'M'
    df1 = df[filterM]
    filterFH= df1["family_history"]=='Yes'
    df2=df1[filterFH]
    count_yes=0
    count_no=0
    i=0
    for index, row in df2.iterrows(): 
        if  row["treatment"]=='Yes':
            count_yes+=1  
        if row["treatment"]=='No':
            count_no+=1
    labels = 'Yes','No'
    sizes = [count_yes,count_no]
    colors = ['r', 'lime']
    explode = (0.1, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Treatment categorization for males with family background history in mental illness")
    plt.axis('equal')
    plt.show()


    # read the data, Data got from Statista
    df = pd.read_csv('survey.csv')
    #Filter gender with female
    filterM = df["Gender"] == 'F'
    df1 = df[filterM]
    background_data = df1['family_history']
    Category=background_data.unique()
    count_yes=0
    count_no=0
    i=0
    for index, row in df1.iterrows(): 
        if  row["family_history"]==Category[0]:
            count_yes+=1  
        if row["family_history"]==Category[1]:
            count_no+=1
    labels = Category[0], Category[1]
    sizes = [count_no,count_yes]
    colors = ['deeppink', 'lightpink']
    explode = (0.1, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Family background history of mental illness in females")
    plt.axis('equal')
    plt.show()
    
    # Mental illness in females with background in family history
    # read the data, Data got from Statista
    df = pd.read_csv('survey.csv')
    filterM = df["Gender"] == 'F'
    df1 = df[filterM]
    filterFH= df1["family_history"]=='Yes'
    df2=df1[filterFH]
    count_yes=0
    count_no=0
    i=0
    for index, row in df2.iterrows(): 
        if  row["treatment"]=='Yes':
            count_yes+=1  
        if row["treatment"]=='No':
            count_no+=1
    labels = 'Yes','No'
    sizes = [count_yes,count_no]
    colors = ['r', 'lime']
    explode = (0.1, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Treatment categorization for females with family background history in mental illness")
    plt.axis('equal')
    plt.show()

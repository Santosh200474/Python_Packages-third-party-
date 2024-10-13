import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

salary_list = pd.read_excel(r"C:\Users\salary_list.xlsx") #Excel sheet in the repository...

#Phase 1(Moment 1): Measures of Central Tendancy and Capturing Center Value
print("Mean of Data")
avg1=salary_list.Rating.mean() #Mean of field: Rating
avg2=salary_list.Salary.mean() #Mean of field: Salary
avg3=salary_list.Salaries_Reported.mean() #Mean of field: Salaries_Reported
print("Mean of field: Rating = ",avg1,"\nMean of field: Salary = ",avg2,"\nMean of field: Salaries_Reported = ",avg3)
print("\n")
print("Median of Data")
med1=salary_list.Rating.median() #Median of field: Rating
med2=salary_list.Salary.median() #Median of field: Salary
med3=salary_list.Salaries_Reported.median() #Median of field: Salaries_Reported
print("Median of field: Rating = ",med1,"\nMedian of field: Salary = ",med2,"\nMedian of field: Salaries_Reported = ",med3)
print("\n")
print("Mode of Data")
mod1=salary_list.Rating.mode() #Mode of field: Rating
mod2=salary_list.Salary.mode() #Mode of field: Salary
mod3=salary_list.Salaries_Reported.mode() #Mode of field: Salaries_Reported
print("Mode of field: Rating = ",mod1,"\nMode of field: Salary = ",mod2,"\nMode of field: Salaries_Reported = ",mod3)
print("\n")
#End of Moment 1

#Moment 2: Capturing Spread/Dispersion
print("Variance of Data")
var1=salary_list.Rating.var() #Variance of field: Rating
var2=salary_list.Salary.var() #Variance of field: Salary
var3=salary_list.Salaries_Reported.var() #Variance of field: Salaries_Reported
print("Variance of field: Rating = ",var1,"\nVariance of field: Salary = ",var2,"\nVariance of field: Salaries_Reported = ",var3)
print("\n")
print("Standard Deviation of Data")
sd1=salary_list.Rating.std() #Deviation of field: Rating
sd2=salary_list.Salary.std() #Deviation of field: Salary
sd3=salary_list.Salaries_Reported.std() #Deviation of field: Salaries_Reported
print("Deviation of field: Rating = ",sd1,"\nDeviation of field: Salary = ",sd2,"\nDeviation of field: Salaries_Reported = ",sd3)
print("\n")
print("Range of Data")
ran1=max(salary_list.Rating)-min(salary_list.Rating) #Range of field: Rating
ran2=max(salary_list.Salary)-min(salary_list.Salary) #Range of field: Salary
ran3=max(salary_list.Salaries_Reported)-min(salary_list.Salaries_Reported) #Range of field: Salaries_Reported
print("Range of field: Rating = ",ran1,"\nRange of field: Salary = ",ran2,"\nRange of field: Salaries_Reported = ",ran3)
print("\n")
#End of Moment 2

#Moment 3: Capturing Skewness shows Direction of Dispersion
print("Skewness of Data")
sk1=salary_list.Rating.skew() #Skewness of field: Rating is Negatively Skewed
sk2=salary_list.Salary.skew() #Skewness of field: Salary is Positively Skewed
sk3=salary_list.Salaries_Reported.skew() #Skewness of field: Salaries_Reported is Positively Skewed
print("Skewness of field: Rating = ",sk1,"\nSkewness of field: Salary = ",sk2,"\nSkewness of field: Salaries_Reported = ",sk3)
print("\n")
#End of Moment 3

#Moment 4: Captures Kurtosis/Peakedness
print("Kurtosis of Data")
kur1=salary_list.Rating.kurt() #Kurtosis of field: Rating is Positive Kurtosis
kur2=salary_list.Salary.kurt() #Kurtosis of field: Salary is Positive Kurtosis
kur3=salary_list.Salaries_Reported.kurt() #Kurtosis of field: Salaries Reported is Positive Kurtosis
print("Kurtosis of field: Rating = ",kur1,"\nKurtosis of field: Salary = ",kur2,"\nKurtosis of field: Salaries_Reported = ",kur3)
print("\n")
#End of Moment 4

#Phase 2: Data Visualization
#Shape Attribute
print(salary_list.shape) #Returns structure of data frame

#Bar Plot: Used to Visualize data but is inconsistent for large amount of data (beyond 100 records)
plt.bar(height = salary_list.Rating, x = np.arange(0,990,1)) #Bar Plot for field: Rating
plt.bar(height = salary_list.Salary, x = np.arange(0,990,1)) #Bar Plot for field: Salary
plt.bar(height = salary_list.Salaries_Reported, x = np.arange(0,990,1)) #Bar Plot for field: Salaries_Reported

#Histogram Plot: Shows Direction of Data Distribution (Also talks about Posibility of Outliers)
plt.hist(salary_list.Rating, color='brown',edgecolor='yellow') #Histogram Plot of field: Rating
plt.hist(salary_list.Salary, bins = [min(salary_list.Salary),salary_list.Salary.quantile(0.528421053),
                                                                       salary_list.Salary.quantile(0.90),
                                                                       salary_list.Salary.quantile(0.98),
                                                                       max(salary_list.Salary)], color='pink',edgecolor='red') #Histogram Plot of field: Salary
plt.hist(salary_list.Salaries_Reported, color='black',edgecolor='white') #Histogram Plot of field: Salaries_Reported

#Box-Plot
plt.figure()
plt.boxplot(salary_list.Rating) #Boxplot of field: Rating (Outliers present at both ends i.e beyound q3(max) and q1(min))
plt.boxplot(salary_list.Salary) #Boxplot of field: Salary (Outliers present at right side i.e beyound q3(max))
plt.boxplot(salary_list.Salaries_Reported) #Boxplot of field: Salaries Reported (Outliers present at right side i.e beyound q3(max))

#Scatter Plot: Talk about Direction and Strength
plt.scatter(x = salary_list["Rating"], y = salary_list["Salary"]) #Strength: Positively Moderate Correlated
plt.scatter(x = salary_list["Rating"], y = salary_list["Salaries_Reported"]) #Strength: Negativeley Moderate Correlated
plt.scatter(x = salary_list["Salary"], y = salary_list["Salaries_Reported"]) #Strength: Weakely Correlated
#salary_list.corr() #Gives Correlation between fields in a matrix form
#End of Phase 2

#Phase 3: Data Cleaning/Cleansing
duplicate=salary_list.duplicated(keep='last') #Returns False where Duplicates exist
duplicate
print("Number of Duplicaetes: ",sum(duplicate)) #Returns Number of Duplicates present in Data Frame
    
#Number of Duplicates found: 0, Hence no need to drop Duplicates

#Handling Ouliers
#Treating Outliers using Winsorization
plt.boxplot(salary_list.Rating) #Before Winsorization  of field: Rating
winsor_iqr1=Winsorizer(capping_method = 'iqr', #Initializing Function
                      tail='both', #cap left, cap right
                      fold=1.5,
                      variables=['Rating']) #Treating outliers in field: Rating

salary_rating=winsor_iqr1.fit_transform(salary_list[['Rating']]) #Function call
plt.boxplot(salary_rating.Rating) #Outliers in field: Rating are removed

#Winzoriation for field: Salary
plt.boxplot(salary_list.Salary) #Before Winsorization
winsor_iqr2=Winsorizer(capping_method = 'iqr', #Initializing Function
                      tail='both', #cap left, cap right
                      fold=1.5,
                      variables=['Salary']) #Treating outliers in field: VOL

salary_sal=winsor_iqr2.fit_transform(salary_list[['Salary']]) #Function call
plt.boxplot(salary_sal.Salary) #After Winsorization: Outliers in field: Salary are removed

#Winzoriation for field: Salaries_Reported
plt.boxplot(salary_list.Salaries_Reported) #Before Winsorization
winsor_iqr3=Winsorizer(capping_method = 'iqr', #Initializing Function
                      tail='both', #cap left, cap right
                      fold=1.5,
                      variables=['Salaries_Reported']) #Treating outliers in field: Salaries Reported

salary_reported=winsor_iqr3.fit_transform(salary_list[['Salaries_Reported']]) #Function call
plt.boxplot(salary_reported.Salaries_Reported) #After Winsorization: Outliers in field: Salaries Reported are removed

#Data Discretization
#Grouping field: Rating into 3 Categories/bins
salary_list["Rating_categories"]=pd.cut(salary_list['Rating'], bins=[min(salary_list.Rating),
                                                                     salary_list.Rating.quantile(0.30),
                                                                     salary_list.Rating.quantile(0.70),
                                                                     max(salary_list.Rating)],
                                        include_lowest=True,
                                        labels=["Low","Mid","High"])
salary_list.Rating_categories.value_counts()

#Grouping of field: Salary into 4 Categories/bins
salary_list["Salary_categories"]=pd.cut(salary_list["Salary"], bins = [min(salary_list.Salary),
                                                                       salary_list.Salary.quantile(0.528421053), #Bin 1: 24K to 5L
                                                                       salary_list.Salary.quantile(0.90), #Bin2: 5L to 11.52L
                                                                       salary_list.Salary.quantile(0.98), #Bin3: 11.52L to 21.22L
                                                                       max(salary_list.Salary)], #Bin4: 21.22L to 38L
                                        include_lowest=True, #including lowest record/Salary
                                        labels = ["Least","Average","Good","Excellent"])

#Data Encoding: Converting Catagorical Data to Numeric Data
#Using OneHotEncoder
sal=OneHotEncoder(drop='first') #initializing method with specifying drop=first, removes first occurance of each category field
salary_list=salary_list[["Rating","Salary","Salaries_Reported",
                        "Company_Name","Job_Title","Location","Employment_Status",
                                     "Job_Roles","Rating_categories","Salary_categories"]]
#Fit and Transform OneHotEncoder
sal_salary_encoder = pd.DataFrame(sal.fit_transform(salary_list.iloc[:,3:]).toarray()) #Obtaining Numerical Data from Categorical data 
#by fitting iloc function onto dataframe and transforming the output to numeric data
dup=sal_salary_encoder.duplicated()
print(dup) #Returns boolean Series of dataframe showing duplicates presence or not
print("\n")
print("Number of Duplicates of Encoded Data: ",sum(dup)) #Getting number of duplictes
drop_salary=sal_salary_encoder.drop_duplicates() #Dropping Duplicates from converted numeric data
drop_salary #Encoded data without Duplicates
print("\n")
#Using Label Encoder
#Initializing Columns to perform encoding
empty_label={}
columns_label=["Company_Name","Job_Title","Location","Employment_Status","Job_Roles","Rating_categories","Salary_categories"] 
for column in columns_label:
    empty_label[column]=LabelEncoder()
    salary_list[column]=empty_label[column].fit_transform(salary_list[column]) #Adding encoded columns onto Original DataFrame: salary_list
  
#Adding Encoded Columns(From OneHotEncoding) onto Original DataFrame: salary_list
salary_list_final_encoded=pd.concat([salary_list,drop_salary],axis=1)

print(salary_list.corr()) #Gives Correlation between fields in a matrix form
#Salary_List contains all the encoded data from LabelEncoder
#LabelEncoder generates numeric values as a singe field per category field.
#Refer salary_list Data Frame for final cleaned and encoded data

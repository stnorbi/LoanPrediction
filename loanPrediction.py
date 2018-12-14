import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os




pd.set_option('display.expand_frame_repr', False)

# File pathes:
sample_path=os.path.dirname(__file__) + "/Samples/"


#Load the dataset into a pandas dataframe
dataFrame=pd.read_csv(sample_path + "train.csv")

print(dataFrame.head())

#Descriptive statistics of numerical fields
desc_stat=dataFrame.describe()
print("\n","Descriptive statistics of numerical fields","\n",desc_stat)


#Frequency analysis of non-numerical variables
def freq_nonnum(df):
    dct={}
    keys=['Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
    for k in keys:
        dct[k]=(df[k].count())

    print("\n","Frequency analysis of non-numerical variables")
    for k,j in dct.items():
        print(k,":",j)

freq_nonnum(dataFrame)

#Distribution analysis of numerical variables
fig, axes=plt.subplots(nrows=3,ncols=3)
fig.tight_layout()
# ax1.set_title("Distribution of Applicant Income")
fig.add_subplot(3,3,1).set_title("Application Income")
dataFrame['ApplicantIncome'].hist(bins=50).plot()


fig.add_subplot(3,3,2)
dataFrame.boxplot(column='ApplicantIncome').plot()


dataFrame.boxplot(column='ApplicantIncome',by='Education', ax=axes[0,2])


plt.subplot(3,3,4).set_title("Loan Amount")
dataFrame['LoanAmount'].plot.hist(bins=50).plot()

dataFrame.boxplot(column="LoanAmount",ax=axes[1,1])

fig.suptitle("")


# Frequency of Credit history
creditHistory=dataFrame["Credit_History"].value_counts(ascending=True)
print("Frequency of Credit History","\n",creditHistory,"\n")

#Probability of getting Loan
getLoan=dataFrame.pivot_table(values='Loan_Status',index=['Credit_History'], aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print("Probability of getting loan:","\n",getLoan)


# Visualization of the pivot tables above
creditHistory.plot(kind="bar",ax=axes[1,2]).set_title("Applicants by Credit History")

getLoan.plot(kind='bar',ax=axes[2,0]).set_title("Probability of getting Loan")


combined_viz=pd.crosstab(dataFrame['Credit_History'], dataFrame['Loan_Status'])
combined_viz.plot(kind='bar',stacked=True,color=['red','blue'],grid=False,ax=axes[2,1]).set_title("Getting Loan by Credit History")

combined_viz2=pd.crosstab(index=[dataFrame['Credit_History'],dataFrame['Gender']], columns=dataFrame['Loan_Status'])
combined_viz2.plot(kind='bar',stacked=True,color=['red','blue'],grid=False,ax=axes[2,2]).set_title("Getting Loan by Gender and Credit History")



print("\n Missing values in data set:\n",dataFrame.apply(lambda x:sum(x.isnull()),axis=0))

print("\n Impute the missing values of LoanAmount by mean:\n")
dataFrame['LoanAmount'].fillna(dataFrame['LoanAmount'].mean(),inplace=True)
print(dataFrame.head(20))

# A key hypothesis is that the whether a person is educated or self-employed can combine to give a good estimate of loan amount.


fig2, axes=plt.subplots(nrows=1,ncols=2)
fig2.tight_layout()
dataFrame.boxplot(column='LoanAmount',by=['Education','Self_Employed'],ax=axes[0])


print("\nImpute the the missing values of Self_Employed variable by 'NO' as its probability is high\n")
dataFrame['Self_Employed'].fillna('No',inplace=True)
print(dataFrame.head(20))


# median tables of "Self_Employed" and "Education" features

table=dataFrame.pivot_table(values="LoanAmount",index='Self_Employed',columns='Education',aggfunc=np.median)

# value print function
def printvalue(x):
    return table.oc[x['Self_Employed'],x['Education']]


# Impute missing values in the LoanAmount feature table

dataFrame['LoanAmount'].fillna(dataFrame[dataFrame['LoanAmount'].isnull()].apply(printvalue,axis=1),inplace=True)
print("\n",dataFrame.head(20))


fig2.suptitle("")
#plt.show()
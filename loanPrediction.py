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
fig=plt.Figure(figsize=(1,4))
ax1=fig.add_subplot(111)
ax1.set_title("Distribution of Applicant Income")
dataFrame['ApplicantIncome'].hist(bins=50).plot


ax2=fig.add_subplot(122)
dataFrame.boxplot(column='ApplicantIncome').plot

dataFrame['LoanAmount'].plot.hist(bins=50)
plt.title("Loan Amount")


dataFrame.boxplot(column="LoanAmount")



dataFrame.boxplot(column='ApplicantIncome',by='Education')


# Frequency of Credit history
creditHistory=dataFrame["Credit_History"].value_counts(ascending=True)
print("Frequency of Credit History","\n",creditHistory,"\n")

#Probability of getting Loan
getLoan=dataFrame.pivot_table(values='Loan_Status',index=['Credit_History'], aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print("Probability of getting loan:","\n",getLoan)


# Visualization of the pivot tables above
fig =plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit History")
creditHistory.plot(kind="bar")

ax2=fig.add_subplot(122)
getLoan.plot(kind='bar')
ax2.set_xlabel("Credit_History")
ax2.set_ylabel("Probability of getting loan")
ax2.set_title("Probability of getting loan by credit history")


# todo correct the crosstab plot!!!
combined_viz=pd.crosstab(dataFrame['Gender'], dataFrame['Loan_Status'],values=dataFrame['Credit_History'],aggfunc=lambda x: x.map({"1.0":1,"0.0":0}).mean())
combined_viz.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)
plt.show()
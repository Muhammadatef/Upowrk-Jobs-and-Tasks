#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install google-search-results')


# ## Here i used serpapi Library it's a library to scrape Google engines with python, the code below describe how i applied the task using this library 
# ## Note : you have to sign in the library website to get your private key : https://serpapi.com/
# 

# In[9]:


## Code to provide searching by keyword : data analyst
from serpapi import GoogleSearch

params = {
  "engine": "google_jobs",
  "q": "data analyst",
  "google_domain": "google.com",
  "location": "Florida, United States",
  "hl": "en",
  "gl": "us",
  "joblink": "https://www.google.com/search?q=google+jobs+data+analyst+florida&sxsrf=ALiCzsY8gEnV7Whr418uakqQ1r7_mo8d4w:1654660845165&ei=7R6gYoDYCYzma_fxktAK&uact=5&oq=google+jobs+data+analyst+florida&gs_lcp=Cgdnd3Mtd2l6EAMyCAghEB4QFhAdOgcIIxCwAxAnOgcIABBHELADOgoIABBHELADEMkDOgcIABCwAxBDOgQIABBDOgUIABCABDoICAAQgAQQyQM6BQgAEJECOgYIABAeEBY6CQgAEB4QyQMQFjoICAAQHhAWEAo6BQghEKABOgcIIRAKEKABSgQIQRgASgQIRhgAUB1Ygh5goB9oAXABeACAAZMBiAH6D5IBBDAuMTeYAQCgAQHIAQrAAQE&sclient=gws-wiz&ibp=htl;jobs&sa=X&ved=2ahUKEwjt_6_D_Jz4AhVlVeUKHRs-CLMQutcGKAF6BAgEEAY#htivrt=jobs&htidocid=s_jdZEaa_okAAAAAAAAAAA%3D%3D&fpstate=tldetail",
  "api_key": "secret_api_key" # you have to put your private key here to work 
}

search = GoogleSearch(params)
results = search.get_dict()


# In[ ]:


## Code to provide searching by keyword : data analytics
from serpapi import GoogleSearch

params = {
  "engine": "google_jobs",
  "q": "data analytics",
  "google_domain": "google.com",
  "location": "Florida, United States",
  "hl": "en",
  "gl": "us",
  "joblink": "https://www.google.com/search?q=google+jobs+data+analyst+florida&sxsrf=ALiCzsY8gEnV7Whr418uakqQ1r7_mo8d4w:1654660845165&ei=7R6gYoDYCYzma_fxktAK&uact=5&oq=google+jobs+data+analyst+florida&gs_lcp=Cgdnd3Mtd2l6EAMyCAghEB4QFhAdOgcIIxCwAxAnOgcIABBHELADOgoIABBHELADEMkDOgcIABCwAxBDOgQIABBDOgUIABCABDoICAAQgAQQyQM6BQgAEJECOgYIABAeEBY6CQgAEB4QyQMQFjoICAAQHhAWEAo6BQghEKABOgcIIRAKEKABSgQIQRgASgQIRhgAUB1Ygh5goB9oAXABeACAAZMBiAH6D5IBBDAuMTeYAQCgAQHIAQrAAQE&sclient=gws-wiz&ibp=htl;jobs&sa=X&ved=2ahUKEwjt_6_D_Jz4AhVlVeUKHRs-CLMQutcGKAF6BAgEEAY#htivrt=jobs&htidocid=s_jdZEaa_okAAAAAAAAAAA%3D%3D&fpstate=tldetail",
  "api_key": "secret_api_key"
}

search = GoogleSearch(params)
results = search.get_dict()


# In[42]:


#here i convert the json file to csv then import the file for the data analyst keyword
import pandas as pd
df = pd.read_csv('A:/Upwork/result_analyst.csv')
df


# In[43]:


df.columns


# In[44]:


#here i convert the json file to csv then import the file  for the data analytics keyword

df1 = pd.read_csv('A:/Upwork/result_analytics.csv')
df1


# In[45]:


df1.columns


# In[46]:


# Here i merged the two tables 
all_data = pd.concat([df,df1])
all_data.info()


# In[47]:


# clean the data 
all_data=all_data.drop(['extensions/0','extensions/1','extensions/2','job_id','thumbnail'], axis=1)


# In[48]:


all_data.head()


# In[49]:


# rename the columns 
all_data.rename(columns = {'detected_extensions/posted_at':'time_job_posted', 'detected_extensions/schedule_type':'job_type','extensions/3':'insurance_or_paid_time_off','extensions/4':'insurance_or_paid_time_off1'}, inplace = True)


# In[50]:


all_data.head()


# In[51]:


# merging two columns into one
all_data["insurance_or_paid_time"] = all_data['insurance_or_paid_time_off1'].astype(str) +"-"+ all_data["insurance_or_paid_time_off"]
all_data.head()


# In[52]:


# drop the two columns
all_data=all_data.drop(["insurance_or_paid_time_off1","insurance_or_paid_time_off"], axis=1)


# In[53]:


all_data.head()


# In[54]:


# replacing the Nan to Unknow in some columns
all_data['insurance_or_paid_time'].replace("NaN", "Unknown",regex=True)
# df['range'] = df['range'].str.replace(',','-')
all_data


# In[40]:


#convert the dataframe into csv file 
all_data.to_csv (r'A:/Upwork\export_dataframe.csv', index = False, header=True)


# ### Thanks for hiring me, looking for more work with you in future

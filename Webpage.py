#!/usr/bin/env python
# coding: utf-8

# In[117]:


pip freeze > requirements.txt


# In[118]:


pip install -r requirements.txt


# ![IMAGE ALT TEXT HERE](http://www1.nyc.gov/assets/home/images/press_release/2019/February/pr101-19.jpg)

# # VISION ZERO - Motor Vehicle Collisions - Crashes

# # Table of Content
# - **Motivation**
# - **Did the COVID-19 Pandemic affect the number of traffic accidents?**
#    - Yearly devolpment
# - **Accident serverity increases during the nigth**
# - **Soft Road Users are more exposed than their counterparties on the road**
# - **The dangers of drunk driving**
#     - For cyclist 
#     - For pedestrians
# - **Failure to Yield Rigth and Driver Awareness**
#     - Failure to Yield Rigth
#     - Driver Awareness
#     - Subconclusion
# - **Does Weather affect number of accidents?**
# - **Conclusion and Recommendations**
# 

# # Motivation

# The city of New York, created a program in 2014 called the Vision Zero. It was created by Mayor of New York City Bill de Blasio. The vision for the program is to eliminate all unnecessary traffic deaths and serious injuries in New York City by 2024. The basis for the program was based on Swedish study, that concluded, that deaths of pedestrian and other soft road users as cyclists in traffic are not as much "accidents" as they are a mere failure of street design.[1]. This notebook as well as Vision Zero have shared goals, as the motivation for Notebook, is to analyze the current situation and improve upon Vision Zero, and helping them achieve their vision. Consequently,  the dataset [Motor Vehicle Collisions from NYC data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95). Each row represents a crash event, which is data going back from 2012 until 2021. Each accidents contains the date and time of the event, as well as who was involved, number of injuries and fatalities, which types of vehicles were invovled, as well as the contribution factors.
# 
# The program has been running for 5-6 years now, and initiatives have been brough about to in order to achive Vision Zero. Among those are lowering of the New York City speed limit from 30mph to 25mph, which was introduced in 2015. Moreover, by advice of Vision Zero advocates in Sweden, an increased amount of speed cameras, together with serveal other laws has been introduced to change the [culture of driving in New York.](https://news.yahoo.com/-vision-zero---one-year-on--assessing-new-york-city-s-quest-to-reduce-preventable-traffic-deaths-174200597.html)
# Among those are increased penalties for failing to yield for pedestrians, bikes or other soft road users, which is considered as a criminal misdemeanor. This is due to increased risk of accidents involving soft road users prior to Vision Zero, and thus pedestrians as cyclists, will be a focus of this project.

# In summary the vision of the this notebook and its inherhent visualzation created, is to amplify the visions of *Vision Zero* by providing the city of New York with tools and insights to help decision makers to determine dangerous causues, vehicles, time intervals, as well as which parts of the city are in need of safer designs, in order to achive *Vision Zero*

# In[91]:


from IPython.display import display, Image
from IPython.display import IFrame


#  

# # Location 

# Firstly a brief overview of where the accidents take place, the number of accidents, and how many soft roaders users where invovled

# In[116]:


IFrame(src='choroplethmap1.html', width=1000, height=700)


# # Did the COVID 19 affect the number of traffic accidents? 

# For the last years and so, the COVID-19 Pandemic has been raging through the world like a step fire. 
# An interestring preliminary analysis, was to see the affect the pandemic have had, on the number of accidents. Moreover, since the beginning of Vision Zero, serveral initiatives have been implemented, and if effective, it should be clear from the number of accidents.

# In[93]:


from IPython.display import display, Image
display(Image(filename='TotalAcci1221.png'))


# As can be seen above the number of accidents has not decreased since the beginning of the oberservation from the dataset, expect in 2020, and 2021, which is due to the COVID-19 having a substantial impact on road users, throughout 2020 and 2021. In 2021, there is only data until april, which explains the low number of accidents. Looking beside the two last years,  the number of accidents has been slowly increasing, which is worrisome, but can also be linked to the number of users on road, since congestions and traffic has increased the last couple of years.  [The more frequently the roads are used, the more likely there is to be an accident](https://ny.curbed.com/2018/2/6/16979696/new-york-city-traffic-congestion-second-worst)
# 
# [The more frequently the roads are used, the more likely there is to be an accident](choroplethmap1.html)
# 

# A more worrisome insigth is that, even though the goal *Vision Zero* was to decrease the number of unnecessary injuries and deaths, it would be eminent that injury and fataility rate dropped as well. Alarming as the plot below demostartes, that the numbers has been growing the last couple of years, even though the number of accidents have been dropping

# In[94]:


display(Image(filename='injfat1221.png'))

##hej##


# From the left the number of injuries relevative to the number of accident are shown. As can be seen, the rate of injury, is somewhat staple, around 25-30%, with a sligth decrease in 2013-15, which the rises again from 2016.
# 
# However, by looking at the two plots, there is an alarming increase of both rate of injury as well as fataility. 
# So even though there has been fewer accidents, the rate of fatatily has increased signifacant, which is worrysome. This could be due to lesser and smaller accidents occuring less frequently, due to less traffic, thus only serious injuries due occur. This, upheld with the contribution factors, this trend is investigated further below:
# Nevertheless by comparing this, with the motivational part where it was stated, there serveal bills and iniatives has been put in place to reduce the injuries and fatalities, it could be argued, that they have not been so effective, as desired. Though the iniatives was mainly focused on certain types of accidents, especially Failure to Yield Rigth, as can been seen below is somewaht decreasing, whereas other factors can be seen rising, which could the reason for the incrased rate of injury/fataility
# 

# In[95]:


IFrame(src='.\yearallcauses.html', width=950, height=700)


# By comparing these two plots, together with the knowlegde of increasing fataility the last couple of years, it can observed that Speeding, play a vital factor in this trend.  By looking on the graph below, this point is further underlined, as speeding is one of the top contributors for fataility.

# In[96]:


IFrame(src='./distibution_causes_all.html', width=1000, height=700)


# Since lesser traffic allow for smoother traffic, it likely that speeding more often occur. Thus, it is important that city of New York continue to figth speeding, as this has been on the rise, and is of the major contributors for fataility. 

# # Accident serverity increases during the night

# [*Do not go gentle into that goodnight* ](https://www.youtube.com/watch?v=3_nprPycZow&ab_channel=JayM) says Michael Caine, telling Matthew McConaughey not accept death passively, as he launches into worm hole in the moive Interstellar, which is believed to be imminent death.
# The same phrase can be somewhat used for New York but maybe with a more safe twist:
# *Do go gentle into that night*, as they people should take extra during the nigth, as the plot below demostrates. The severity of accidents grows substantially during the nigth early hours of the morning.

# In[97]:


display(Image(filename='injfathourly.png'))


# # Soft Road Users are more exposed to accidents than counterparties on the road

# As stated from motivional part, soft road user are one of the main focuses, on this notebook, which became imminent when investigating how the soft roads users as bikes and pedestrians are involved in accident. 
# They face a higher risk, especially counterparty risk, i.e. where other drivers causes harm to them, which is shown in the plot below

# In[98]:


IFrame(src='./injury_fatal5.html', width=950, height=700)


# In[99]:


IFrame(src='./injury_fatal3.html', width=950, height=700)


# An interstring takeaway from the plot above is, when focusing on the level on injury of both pedestrians and cyclists compared with overall distribution. Around 80% of all the accidents involving soft road users, results in injury, which a factor 5 times compared with the overall distribution. Moreover, by comparing fatality, it can be observed, that the frequency is relatively higher than the overall, luckly still low. 
# Based on these numbers, the main causes of contribution factors where investigated, to indentify any pitfalls for pedestrians and cyclists, and what contributes to the high level of accidents

# In[100]:


IFrame(src='./causes_frequency_datasets.html', width=1000, height=700)


# From the plot above, it is quite clear, that especially factors as Failure to Yield Rigth, as Driver Inattion, causes alot of accidents, in particular for cyclist and pedestrians, as these are more present, than the average number of accident in general. Consequently, these are contribution factors to accidents that are investigated in further details.

# # Failure to Yield Rigth and Driver Awareness

# As can be seen from the plot above, driver distractions and failure to yield right are common accidents occurring for pedestrians and cyclist, thus putting them in huge counterparty risk, especially for pedestrians, as these two factors contribute to almost 50% of all of accidents.  For cyclist, these two factors contribute in approximately 35% of the accidents. 
# For ‘Failure to yield right’ it is relevant to see which crosses are represented most times in the accident statistics. The most represented crosses would be places of interest for further inspection to see if street design could be improved in the crosses to prevent future accidents.
# Similarly, the ‘Driver Inattention/Distraction’ factor is also relevant to investigate further to see if any locations distract drivers more often.
# An analysis of the most common locations for these two contributing factors are carried out based on accidents involving a pedestrian or a cyclist.

# ### Failure to Yield Rigth 
# By knowing which causes are the main contributors putting bikes and pedestrians in exstensive risk, this knowlegde can be utilized to provide a map, of which intersections are in need of an update, in terms of design or more awareness among drivers:
# 

# In[101]:


IFrame(src='./worst_yield_right.html', width=1000, height=700)


# As can be seen there are serveral hotspots, from which it accidents more often occur, like in Queens around flushing main street, where a lot of failure to yield rigth accidents occur. Also around Brooklyn, Tillary Street and Myrtle street alot of accidents too occur. 
# These areas should be investiaged, by the city of New York to determine why drivers fail to yield rigth, and how it could be improved, as these are potential hot spots for accidents, that could have been avoided.

# ### Driver distraction 
# 
# 

# In[102]:


IFrame(src='./worst_driver_distraction.html', width=1000, height=700)


# As can be seen from driver distraction, the most dangerous parts of town for pedestrians and cyclist are mostly located wihtin Manhatten, but also Brooklyb, but especially in the exit around Queensboro bridge, where alot of distractions causes accidents. 
# Another area is around Noho and China town, and especially the cross of Kenmare Street and Chrystie street is a dangerous intersection, as this section is both high in Driver Distraction and Failure to Yield Rigth 
# 

# ### Subconclusion 
# Both causes can be seen on the maps, to be centered around different spots throughout the city. This can then be utilizied, by the city of New York to create awareness of the danger of these areas and intersection. Though it could be argueded, that failure to yield rigth, and driver distractions is subjective to the driver, the maps above demostrates, that certain areas and intersections are more prone to accidents than other. This underlines the fact that accidents due occur due to failure of street design, which then can be used by the city of New York to update their current street design. 

# # Other relevent Contribution Factors for Accidents 

# As stated two major factors play a vital part in terms of accidents for soft road users. 
# A interstring perspective was to investigate if any of these accidents, where more prone to occur during certain peak hours. As the plots demostrating below, certain accidents, are more likely to happen during certain hours, some of which make untuitively sense, whereas others requires a deeper dive

# In[103]:


IFrame(src='./hour_pedestrian_causes.html', width=950, height=700)


# In[104]:


IFrame(src='./hour_cyclist_causes.html', width=950, height=700)


# From the plots there are certain key point of interst. One of the being that *Pavement Slippery* mostly occurs during sunrise and sunset, which makes sense, but is still quite interstring. Same tendency is observed for *Glare*, as this mostly happen during the morning hours. Also Driver Inatention, can be seen to have certain peak hours, as they mainly happen during the afternoon where people get of work. Thus it important, that the people remain focused in the traffic, as more accident are likely to happen during these hours, which is one of the major contributors for soft road accidents.
# A very interstring takeaway is accident happening due to *Alcohol Involvment*, as this mainly happens during the nigth, and not as much during the day. Rembering the plot demostrating rate of injury and fataility, as injuries and fatailites have higher rates as well during the night, thus indicating, that there could be a correlation between Drunk Driving and the high fataility during the night, which will be investigated further

# # The Dangers of Drunk Driving

# Drunk driving is known cause of accidents in New York, and something the city, and US in general have tried to cut down on. This section will investiage, how the devolopment of drunk driving has been the last couple of years, as well non volentary inidivduals being effected by drunk driving. Does the accidents only cause self harm, or do the accidents cause harm to others as well? 
# 
# 

# ## For cyclists

# In[105]:


IFrame(src='./distibution_causes_cyclist.html', width=1000, height=700)


# One of the reasons for alcohol invovolment, is an interestring topic, is due to its high fataility rate. As can be seen on the plot above, it is the most fatal type of accidents, causing a fataility in roughly 5% of the accidents, and injury in 80%. Luckily it only contributes for around 1% of the accidents, but it is still af factor that needs to be mitigated. As stated cyclist and pedestrins are in greater risk of counterparty failure or accidents, thus the following plots illustrates the cotribution factors for cyclist over a 24 hour span. From the plot it can be derived, that accidents occuring to cyclists, with the highest rates of fataility as alcohol invovlment, are most common during the nigth, as well as during the weekend.

# In[106]:


IFrame(src='./hour_cyclist_causes.html', width=950, height=700)


# In[107]:


IFrame(src='./weekday_cyclist_causes.html', width=950, height=700)


# So accidents for cyclist involving drunk driving are most commonly during the nigth, as well during the weekend, though Wednesday has the highest rate of fataility. Consequetly it would be expected that cyclists are in most danger of being killed in traffic, during the weekend nights: 

# In[108]:


display(Image(filename='injfathourlypdcy1.png'))
display(Image(filename='injfatdaycy.png'))


# As expected cyclist are more prone to getting killed in the traffic during the nigth and early hours of the morning, with quite alarming figures that ranges from 2-5 time more likely. Also when cyclist are involved in an accident, it is almost with a 100% certainty, that they are getting injured, putting them at an exstensive risk. 
# Thus it can be derived from figures, that cyclist are more prone to fataility during the nigth, which often is linked to alcohol involvement.

# ## For pedestrians

# In[109]:


IFrame(src='./distibution_causes_pedestrians.html', width=1000, height=700)


# The same tendency is clear for pedestrians, but only intensified. Alchohol involvment is the major contributor for pedestrian fatalities, which must be adressed in order to reduce the number of fatalities. By comparison, ancidents invovling alcohol only contribute to roughly 1%, yet of all accidents for both pedestians as cyclist they are the most fatal, thus making an important factor to mitigate, and decrease, in order to avoid unnecessary fatailities. 
# Moreover traffic control disregard, and speeding are some of the major factors for fatalities, and these factors should be mitigated as well in order to avoid any further unnecessary accidents. 
# 
# Moreover as was seen from accidents involving cyclists, alcohol accidents affecting pedestrians, does also mostly take place during the night, as well as during the weekend, as the figure below demostrate. In terms of speeding and disobyeing traffic controls, those factors occur more often durig the day, as well as during the midweek. 

# In[110]:


IFrame(src='./hour_pedestrian_causes.html', width=950, height=700)


# In[111]:


IFrame(src='./weekday_pedestrian_causes.html', width=950, height=700)


# As can be seen from the plots, accidents involving pedestrians are most likely to occur during late afternoon night and especially in the weekend Thus pedestrians have increased risk of being involved in an accident during those hours, and are in more danger of fatality, which is underlined in the plot below. Moreover it can be observe, that almost every time a pedestrian is invovled in an accident, no matter the hour, it almost always results in injury. 

# In[112]:


display(Image(filename='injfathourlypdcy.png'))
display(Image(filename='injfatdaypd.png'))


# ## Summary
# 
# Drunk Driving  and rate of accidents for bike and pedestrians are heavely linked, as drunk driving is one main contributors to fataility among soft road users, and mostly takes place during the nigth. 
# This was especially a problem in the beginning of 2016, as can be seen from the figures above, which has luckly declined since, which has luckly decrease since. This is throughout the years, which can partly be due to their sobriety-checkpoints, and their [increased focus on drunk driving](https://www.cdc.gov/motorvehiclesafety/pdf/impaired_driving/Drunk_Driving_in_NY.pdf)
# 
# 
# 

# # Does weather influence accidents? 

# Another interstring point derived from the data, was that by comparing the data with weather data of New York from, it was clesar, that certain weather conditions do play a part in terms of accidents. Diffucult weather conditions, results in more injuries and fatailities, but only for pedestrians. 
# 
# Interestringly enough, the amount of accidents, injuries or deaths, does not increase for cyclist as a consequence of weather conditions, which was the initial thoughth. Most accidents occur in sunny weather with plus degress. By comparing data it was observed that accidents on bikes occur mostly in the time spans with good weather, during the summers, thus explaining, why accidents where more serve in good weather condition. People simply did not bike during the winter. This intuitively is logic, since riding a bike in the winther or during bad weather is unpleasent, and is often avoided.
# For pedestrians on the other hand, it the tendency was clear as shown below:

# In[113]:


display(Image(filename='weather_inj_fat_rateped.png'))


# From this graph it can be observed, that weather conditions do play a vital part in when pedestrians are most likely to be injured or killed in traffic. As demostrated above, even though *dry roads* is the most common of weather conditions for accidents, injuries and deaths, the most dangerous conditions in terms of injuries, are wet and snowy roads, as these pose a great prone to injruies among pedestrians. In terms of fatalities, heavy wet road, is the most dangerous, with a average fataility per day of 0.2, though closely followed by dry road +. 
# In terms of contribution reasons it can also be observed, that changing weather conditions, causes more accidents in particular within the major contribution factors as Failure to Yield Rigth and Driver Distraction, as shown below.

# In[114]:


IFrame(src='./avg_causes_frequency_weather.html', width=950, height=700)


# # Conclusion & Recommendation 

# ‘Failure to yield right’ and ‘Driver Inattention/Distraction’ are contributing factors for nearly half of accidents involving pedestrians.  35 % for cyclist accidents.Most accidents happen during peak hoursAccidents happening at night more often than during the day have a fatal outcomeAlcohol involvement are a factor with high frequency of fatal outcomes for accidentsAlcohol involvement mostly happens during the night and during weekendsFor pedestrian the number of accidents are fairly constant throughout the year whereas for the number for cyclist drops in winter periods – most likely due to fewer people cycling

# ## Conclusion 
# 
# - The COVID-19 pandemic have had an impact of the number of accidents occuring, but the rate of injury and fatality has been increasing
#     - This is partly due to the increase of accidents happening due to speeding, which has a high rates of injury and fataility 
# - The city of New York efforts to reduce unnecessary injuries and accidents has to some extend been effective as the Failure to Yield Right has been decreasing, though as mentioned, speeding has increased
# - Soft road users as bikes and pedestrians are more prone to accidents, and when involved, they are often associcated with higher risk of injury and fatality, than counterparties, like vehicles. 
#     - Around 80% of all the accidents involving soft road users, results in injury, which a factor of 5, compared with the overall distribution.
#     
# - There are substantially higher rates of fataility among soft road users during certain peaks hours, especially at night.
#     - Mainly due to high number of accidents invovling alcohol, which is the predominant cause of the high fatality rates during the night. 
#          - Luckily that state of New York has been cutting down on the number of accidents in the last couple of years. 
#     - Unsafe speed (speeding), is also a contribution factor to the high level of injury and fataility, which has been increasing under the COVID-19 pandemic 
# - Driver distractions and and Failure to Yield Rigth is the predominant contribution factors to cyclist and pedestria accident, thus putting them in huge counterparty risk. 
#      - Almost 50% of accidents invovling pedestrians are due to driver distractions and and failure to yield rigth
#      - Around 35% of accidents invovling cyclist are due to driver distractions and and failure to yield rigth
# - Weather conditions does play a part, in terms of severity of accidents, but mostly for pedestrains, as the more rain and snows causes higer rates of fataility. 
# 
# 
# As stated in the motivational part the motivation for this Notebook, was to analyze the current situation and improve upon Vision Zero, and helping them achieve their vision.
# From the analysis conducted, the notebook is now able to provide detailed information and insigths, to where and when  the city of New York should allocate more resources in order to achive *Vision Zero* 
# 
# ## Recommendation
# 
# Remebering the motivation for the project, was to provide an overview, where, when and why accidents occured, and moreover to provide the city of New York with a tools and insights to utilized to update current street design. 
# By using the knowlegde of the contribution factors, when accidents take place, which this is then further linked, with where these accidents takes place the following recommendation are provided:
# 
# 
# **Soft Road Users are more exposed to accidents than counterparties on the road**
# 
# Soft road users as cyclist and pedestrians, are as expressed more exposed to accidents, with higher frequency and when involved, they are often associcated with higher risk of injury and fatality, than counterparties. Thus from the analysis, the board of Vision Zero should move forward with initiatives that recudes the risk of injury of being a pedestrian or cyclist in New York, as many of these accidents are avoidable, and are based on counterparty failures. 
# 
# **Failure to Yield Rigth and Driver Awareness** 
# 
# As stated this these factors play a major part in the number of accidents invovling soft road users, and needs to be mitigated. From the analysis, it was both derived, when these accidents mostly occur, but most importantly where they occur. By using the knowlegde of the contribution factors, this is then further linked, with where these accidents takes place, which then will be pint pointed out, thus providing the where, to which street designs should be updated. The plots maps provided, can be used as guidelines and indicators, where and when ressources should be allocate in order to reduce the number of accidents
# 
# **The Dangers of Drunk Driving** 
# 
# Alchohol involvment is the major contributor for cyclist and pedestrian fatalities, which must be adressed in order to reduce the number of fatalities. Luckily by comparison, ancidents invovling alcohol only contribute to roughly 1%, yet of all accidents, for both pedestians as cyclist, they are the most fatal, thus making an important factor to mitigate, and decrease, in order to avoid unnecessary fatailities. Moreover traffic control disregard, and speeding are some of the major factors for fatalities, and these factors should also be mitigated as well in order to avoid any further unnecessary accidents.
# 
# The consumption of alcohol and rate of accidents for bike and pedestrians, was especially a problem in the beginning of 2016,  which has luckly declined since. This is can partly be due to their sobriety-checkpoints, and their [increased focus on drunk driving](https://www.cdc.gov/motorvehiclesafety/pdf/impaired_driving/Drunk_Driving_in_NY.pdf)
#  [4]. 
# 

# In[ ]:





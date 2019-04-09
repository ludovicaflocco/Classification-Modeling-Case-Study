#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Thu Oct 18 20:00:00 2018

@author: Marialudovica Flocco 

Working Directory: /Users/ludovicaflocco/Desktop/Machine_Learning/GoT

Purpose: 
    1. Exploring the Game of Thrones dataset 
    2. Building an accurate prediction model for the death of the characters

AGENDA:
    1)  Initial Data Set-up
    2)  Data Exploration
    3)  Data Cleaning 
    4)  Treatment of Missing Values
    5)  Outlier Treatment
    6)  Grouping Data
    7)  Correlation Analysis
    8)  Factorization & Dummy Variables
    9)  OLS (Ordinary Least Square) Method Full Model
    10) KNN MODEL
    11) Cross Validation
    12) Scikit Learn LR Significant Model


"""

###############################################################################
# Initial Data Set-up
###############################################################################

# Loading Libraries
import pandas as pd
import statsmodels.formula.api as smf # regression modeling
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter # counting

# For KNN model
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.metrics import roc_auc_score # AUC score
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.model_selection import GridSearchCV # hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LinearRegression

###############################################################################

file ='GOT_character_predictions.xlsx'
GoT = pd.read_excel(file)


###############################################################################
# Data Exploration 
###############################################################################


# Dimensions of the DataFrame
GoT.shape #26 variables, 1946 observations

# Information about each variable 
print(GoT.info())

# Basic Description:
print(GoT.describe())

# Extracting column names
GoT.columns
#S.NO = Character number (by order of appearance)
#name = Character name
#title = Honorary title(s) given to each character
#male -> 1 = male, 0 = female
#culture	 = Indicates the cultural group of a character
#dateOfBirth	= Known dates of birth for each character (measurement unknown)
#mother	= Character's biological mother
#father	= Character's biological father
#heir = Character's biological heir
#house = Indicates a character's allegiance to a house (i.e.a powerful family)
#spouse = Character's spouse(s)
#book1_A_Game_Of_Thrones -> 1 = appeared in book, 0 = did not appear in book
#book2_A_Clash_Of_Kings -> 1 = appeared in book, 0 = did not appear in book
#book3_A_Storm_Of_Swords -> 1 = appeared in book, 0 = did not appear in book
#book4_A_Feast_For_Crows -> 1 = appeared in book, 0 = did not appear in book
#book5_A_Dance_with_Dragons ->1 = appeared in book, 0 = did not appear in book
#isAliveMother -> 1 = alive, 0 = not alive 
#isAliveFather -> 1 = alive, 0 = not alive
#isAliveHeir -> 1 = alive, 0 = not alive
#isAliveSpouse -> 1 = alive, 0 = not alive
#isMarried -> 1 = married, 0 = not married
#isNoble -> 1 = noble, 0 = not noble
#age	-> Character's age in years
#numDeadRelations -> Tot n. of deceased relatives throughout all of the books
#popularity -> (1 = extremely popular (max), 0 = extremely unpopular (min))
#isAlive -> 1 = alive, 0 = not alive


# Total number of missing values:
na_total_i = GoT.isnull().sum().sum()

# Total number of missing values per variable:
print(
      GoT
      .isnull()
      .sum()
      )



###############################################################################
# Data Cleaning
###############################################################################


###############################################################################
# Deliting Variables
###############################################################################

# Deleting S.No:
del(GoT['S.No'])
# Unrelevant Variable for our prediction Model

# Deleting name:
del(GoT['name'])
# Unrelevant variable for our prediction Model 

# Deleting dateOfBirth:
del(GoT['dateOfBirth'])
# Measurement of date of Birth is unknown therefore not reliable 
# Moreover the column "Age" makes this variable not required 

# Deleting mother:
del(GoT['mother'])
# Only 21 out of 1946 observations 
# No possible way to impute or assume missing values 

# Deleting father:
del(GoT['father'])
# Only 26 out of 1946 observations 
# No possible way to impute or assume missing values 

# Deleting heir:
del(GoT['heir'])
# Only 23 out of 1946 observations 
# No possible way to impute or assume missing values 

# Deleting spouse:
del(GoT['spouse'])
# Unrelevant variable for our prediction Model 

# Deleting isAliveMother:
del(GoT['isAliveMother'])
# Only 21 out of 1946 observations. 
# No possible way to impute or assume missing values 

# Deleting isAliveFather:
del(GoT['isAliveFather'])
# Only 26 out of 1946 observations available. 
# No possible way to impute or assume missing values 

# Deleting isAliveHeir:
del(GoT['isAliveHeir'])
# Only 23 observations out of 1946 
# No possible way to impute or assume missing values 

# Deleting isAliveSpouse:
del(GoT['isAliveSpouse'])
# Only 276 observations out of 1946
# No possible way to impute or assume missing values 

###############################################################################
# Treatment of Missing Values
###############################################################################

###################################
# Flagging 
###################################

for col in GoT:
    if GoT[col].isnull().any():
        GoT['missing_'+col] = GoT[col].isnull().astype(int)
        
###################################
# Imputing 
###################################
        
# Title Column
fill_title = "Unknown_title"
GoT["title"] = GoT["title"].fillna(fill_title)


# Culture Column
fill_culture = "Unknown_culture"
GoT["culture"] = GoT["culture"].fillna(fill_culture)


# House Column
fill_house = "Unknown_house"
GoT["house"] = GoT["house"].fillna(fill_house)


# Age Column
fill_age = -1
GoT["age"] = GoT["age"].fillna(fill_age)
# I chose a negative number so that it will allow me to analyze 
# But at the same time it will not interfear with my data 
# -1 = No age mentioned

###############################################################################
# Outlier Treatment
###############################################################################

###################################
# Age
###################################

# Treat outliers like NAs:
GoT.loc[110, 'age'] = -1
GoT.loc[1350, 'age'] = -1

###############################################################################
# Grouping Data
###############################################################################

###################################
# Title
###################################

# Lower Case:
GoT['title'] = GoT['title'].str.lower()

# Keeping only the first "word":
GoT["title"] = GoT["title"].apply(lambda x: x.split()[0])

# Grouping titles:
GoT["title"] = GoT["title"].replace(
        {"captain":"military",
         'castellancommander':'military',
         'castellan':'military',
         'east-watch-by-the-sea':'military',
         'khal':'military',
         'khalakka':'military',
         'Khalko':'military',
         'chief':'military',
         'knight':'military',
         'master-at-arms':'military',
         'sercastellan':'military',
         'ser':'military',
         'hand':'lord-or-lady',
         'lady':'lord-or-lady',
         'ladyqueen':'lady-or-lord',
         'ladyqueendowagen':'lady-or-lord',
         'lord':'lord-or-lady',
         'lordsport':'lord-or-lady',
         'lordwisdom':'lord-or-lady',
         'grand':'maester',
         'maester':'maester',
         'magister':'maester',
         'master':'maester',
         'wisdom':'maester',
         'archmaester':'maester',
         'oarmaster':'maester',
         'king-beyond-the-wall':'royal',
         'king':'royal',
         'prince':'royal',
         'princess':'royal',
         'princessqueen':'royal',
         'princesssepta':'royal',
         'protector':'royal',
         'godsgrace':'royal',
         'queen':'royal',
         'queenblack':'royal',
         'queendowagen':'royal',
         'septa':'religious-title',
         'septon':'religious-title'})

# Other title
GoT.loc[GoT['title'].value_counts()[GoT['title']].values < 10, 'title'] = "other_title"

###################################
# Culture
###################################

# Lower Case for Culture:
GoT['culture'] = GoT['culture'].str.lower()

# Keeping only the first "word":
GoT["culture"] = GoT["culture"].apply(lambda x: x.split()[0])

# Grouping cultures:
GoT["culture"] = GoT["culture"].replace(
        {'andal':'free-cities-across-the-narrow-sea',
         'andals':'free-cities-across-the-narrow-sea',
         'braavos':'free-cities-across-the-narrow-sea',
         'braavosi':'free-cities-across-the-narrow-sea',
         'lysene':'free-cities-across-the-narrow-sea',
         'lyseni':'free-cities-across-the-narrow-sea',
         'tyroshi':'free-cities-across-the-narrow-sea',
         'myrish':'free-cities-across-the-narrow-sea',
         'pentoshi':'free-cities-across-the-narrow-sea',
         'norvos':'free-cities-across-the-narrow-sea',
         'norvoshi':'free-cities-across-the-narrow-sea',
         'qohor':'free-cities-across-the-narrow-sea',
         'dorne':'dornish',
         'dornish':'dornish',
         'dornishmen':'dornish',
         'ghiscaricari':'ghiscari',
         'ironmen':'ironborn',
         'ironborn':'ironborn',
         'riverlands':'rivermen',
         'rivermen':'rivermen',
         'astapor':'valyrian-peinsula',
         'astapori':'valyrian-peinsula',
         'meereen':'valyrian-peinsula',
         'meereenese':'valyrian-peinsula',
         'valyrian':'valyrian-peninsula',
         'northern mountain clans':'northmen',
         'northmen':'northmen', 
         'reach':'reach',
         'reachmen':'reach',
         'the Reach':'reach',
         'vale':'valemen',
         'vale mountain clans':'valemen',
         'valemen':'valemen',
         'westerlands':'westermen',
         'westerman':'westermen',
         'westermen':'westermen',
         'westeros':'westermen',
         'westermen':'westermen',
         'wildling':'free_folk',
         'wildlings':'free_folk',
         'free':'free_folk'})        
    
# Other Culture
GoT.loc[GoT['culture'].value_counts()[GoT['culture']].values < 10, 'culture'] = "other_culture"
    
###################################
# House
###################################

# Lower Case
GoT['house'] = GoT['house'].str.lower()

###################################
# Grouping Houses by geographical region
###################################
GoT["house"] = GoT["house"].replace(
        {'house bolton':'houses from the north',# Houses from the North
         'house bolton of the dreadfort':'houses from the north',
         'house cassel':'houses from the north',
         'house cerwyn':'houses from the north',
         'house condon':'houses from the north',
         'house dustin':'houses from the north',
         'house flint':'houses from the north',
         "house flint of Widow's Watch":'houses from the north',
         'house glover':'houses from the north',
         'house harclay':'houses from the north',
         'house hornwood':'houses from the north',
         'house karstark':'houses from the north',
         'house liddle':'houses from the north',
         'house locke':'houses from the north',
         'house manderly':'houses from the north',
         'house mollen':'houses from the north',
         'house mormont':'houses from the north',
         'house norrey':'houses from the north',
         'house poole':'houses from the north',
         'house reed':'houses from the north',
         'house ryswell':'houses from the north',
         'house stout':'houses from the north',
         'house tallhart':'houses from the north',
         'house umber':'houses from the north',
         'house woods':'houses from the north',
         'house woolfield':'houses from the north',
         'house wull':'houses from the north',
         'house allyrion':'houses from dorne',# Houses from Dorne
         'house blackmont':'houses from dorne',
         'house dalt':'houses from dorne',
         'house dayne':'houses from dorne',
         'house dayne of high hermitage':'houses from dorne',
         'house drinkwater':'houses from dorne',
         'house fowler':'houses from dorne',
         'house gargalen':'houses from dorne',
         'house jordayne':'houses from dorne',
         'house manwoody':'houses from dorne',
         'house qorgyle':'houses from dorne',
         'house santagar':'houses from dorne',
         'house toland':'houses from dorne',
         'house uller':'houses from dorne',
         'house vaith':'houses from dorne',
         'house Yronwood':'houses from dorne',
         'house ambrose':'houses from the reach',# Houses from the Reach 
         'house ashford':'houses from the reach',
         'house ball':'houses from the reach',
         'house beesbury':'houses from the reach',
         'house blackbar':'houses from the reach',
         'house bulwer':'houses from the reach',
         'house bushy':'houses from the reach',
         'house caswell':'houses from the reach',
         'house chester':'houses from the reach',
         'house cockshaw':'houses from the reach',
         'house conklyn':'houses from the reach',
         'house costayne':'houses from the reach',
         'house crane':'houses from the reach',
         'house cuy':'houses from the reach',
         'house florent':'houses from the reach',
         'house fossoway':'houses from the reach',
         'house fossoway of Cider Hal':'houses from the reach',
         'house graceford':'houses from the reach',
         'house grimm':'houses from the reach',
         'house hewett':'houses from the reach',
         'house hightower':'houses from the reach',
         'house hunt':'houses from the reach',
         'house inchfield':'houses from the reach',
         'house leygood':'houses from the reach',
         'house meadows':'houses from the reach',
         'house merryweather':'houses from the reach',
         'house mullendore':'houses from the reach',
         'house norcross':'houses from the reach',
         'house norridge':'houses from the reach',
         'house osgrey':'houses from the reach',
         'house peake':'houses from the reach',
         'house redwyne':'houses from the reach',
         'house rhysling':'houses from the reach',
         'house risley':'houses from the reach',
         'house rowan':'houses from the reach',
         'house serry':'houses from the reach',
         'house tarly':'houses from the reach',
         'house uffering':'houses from the reach',
         'house varner':'houses from the reach',
         'house vyrwel':'houses from the reach',
         'house webber':'houses from the reach',
         'house willum':'houses from the reach',
         'house wythers':'houses from the reach',
         'house arryn':'houses from the vale',# Houses from the Vale
         'house baelish':'houses from the vale',
         'house belmore':'houses from the vale',
         'house borrell':'houses from the vale',
         'house coldwater':'houses from the vale',
         'house corbray':'houses from the vale',
         'house egen':'houses from the vale',
         'house grafton':'houses from the vale',
         'house hardyng':'houses from the vale',
         'house hunter':'houses from the vale',
         'house longthorpe':'houses from the vale',
         'house lynderly':'houses from the vale',
         'house moore':'houses from the vale',
         'house redfort':'houses from the vale',
         'house royce':'houses from the vale',
         'house royce of the gates of the moon':'houses from the vale',
         'house shett of gull tower':'houses from the vale',
         'house sunderland':'houses from the vale',
         'house templeton':'houses from the vale',
         'house tollett':'houses from the vale',
         'house waynwood':'houses from the vale',
         'house banefort':'houses from westerlands',# Houses from Westerlands
         'house bettley':'houses from westerlands',
         'house brax':'houses from westerlands',
         'house broom':'houses from westerlands',
         'house clegane':'houses from westerlands', 
         'house clifton':'houses from westerlands',
         'house crakehall':'houses from westerlands',
         'house estren':'houses from westerlands',
         'house farman':'houses from westerlands',
         'house foote':'houses from westerlands',
         'house greenfield':'houses from westerlands',
         'house hetherspoon':'houses from westerlands',
         'house jast':'houses from westerlands',
         'house kenning of kayce':'houses from westerlands',
         'house lefford':'houses from westerlands',
         'house lorch':'houses from westerlands',
         'house lydden':'houses from westerlands',
         'house marbrand':'houses from westerlands',
         'house moreland':'houses from westerlands',
         'house payne':'houses from westerlands',
         'house peckledon':'houses from westerlands',
         'house plumm':'houses from westerlands',
         'house prester':'houses from westerlands',
         'house reyne':'houses from westerlands',
         'house ruttiger':'houses from westerlands',
         'house sarsfield':'houses from westerlands',
         'house spicer':'houses from westerlands',
         'house stackspear':'houses from westerlands',
         'house swyft':'houses from westerlands',
         'house turnberry':'houses from westerlands',
         'house vikary':'houses from westerlands',
         'house westerling':'houses from westerlands',
         'house yarwyck':'houses from westerlands',
         'house yew':'houses from westerlands',
         'house blacktyde':'houses from iron islands',# from the Iron Islands  
         'house botley':'houses from iron islands',
         'house codd':'houses from iron islands',
         'house drumm':'houses from iron islands',
         'house farwynd':'houses from iron islands',
         'house farwynd of lonely light':'houses from iron islands',
         'house goodbrother':'houses from iron islands',
         'house goodbrother of shatterstone':'houses from iron islands',
         'house harlaw':'houses from iron islands',
         'house harlaw of grey garden':'houses from iron islands',
         'house harlaw of harlaw hall':'houses from iron islands',
         'house harlaw of harridan hill':'houses from iron islands',
         'house harlaw of the tower of glimmering':'houses from iron islands',
         'house hoare':'houses from iron islands',
         'house humble':'houses from iron islands',
         'house ironmaker':'houses from iron islands',
         'house kenning of harlaw':'houses from iron islands',
         'house merlyn':'houses from iron islands',
         'house myre':'houses from iron islands',
         'house oakheart':'houses from iron islands',
         'house sharp':'houses from iron islands',
         'house shepherd':'houses from iron islands',
         'house sparr':'houses from iron islands',
         'house tawney':'houses from iron islands',
         'house volmark':'houses from iron islands',
         'house weaver':'houses from iron islands',
         'house wynch':'houses from iron islands',
         'house blackwood':'houses from riverlands',#from the Riverlands 
         'house blanetree':'houses from riverlands',
         'house bracken':'houses from riverlands',
         'house butterwell':'houses from riverlands',
         'house charlton':'houses from riverlands',
         'house cox':'houses from riverlands',
         'house darry':'houses from riverlands',
         'house deddings':'houses from riverlands',
         'house erenford':'houses from riverlands',
         'house goodbrook':'houses from riverlands',
         'house grell':'houses from riverlands',
         'house haigh':'houses from riverlands',
         'house hawick':'houses from riverlands',
         'house heddle':'houses from riverlands',
         'house lothston':'houses from riverlands',
         'house lychester':'houses from riverlands',
         'house mallister':'houses from riverlands',
         'house mooton':'houses from riverlands',
         'house mudd':'houses from riverlands',
         'house nayland':'houses from riverlands',
         'house paege':'houses from riverlands',
         'house pemford':'houses from riverlands',
         'house piper':'houses from riverlands',
         'house roote':'houses from riverlands',
         'house ryger':'houses from riverlands',
         'house smallwood':'houses from riverlands',
         'house stonehouse':'houses from riverlands',
         'house stonetree':'houses from riverlands',
         'house strong':'houses from riverlands',
         'house tully':'houses from riverlands',
         'house vance':'houses from riverlands',
         'house vance of atranta':'houses from riverlands',
         "house vance of wayfarer's Rest":'houses from riverlands',
         'house vypren':'houses from riverlands',
         'house wayn':'houses from riverlands',
         'house whent':'houses from riverlands',
         'house wodehouse':'houses from riverlands',
         'house blount':'houses from crownlands',# Houses from the Crownlands 
         'house boogs':'houses from crownlands',
         'house brune of brownhollow':'houses from crownlands',
         'house brune of the dyre den':'houses from crownlands',
         'house buckwell':'houses from crownlands',
         'house byrch':'houses from crownlands',
         'house bywater':'houses from crownlands',
         'house celtigar':'houses from crownlands',
         'house chelsted':'houses from crownlands',
         'house chyttering':'houses from crownlands',
         'house crabb':'houses from crownlands',
         'house darklyn':'houses from crownlands',
         'house farring':'houses from crownlands',
         'house gaunt':'houses from crownlands',
         'house hardy':'houses from crownlands',
         'house hayford':'houses from crownlands',
         'house hogg':'houses from crownlands',
         'house hollard':'houses from crownlands',
         'house kettleblack':'houses from crownlands',
         'house longwaters':'houses from crownlands',
         'house mallery':'houses from crownlands',
         'house massey':'houses from crownlands',
         'house rambton':'houses from crownlands',
         'house rosby':'houses from crownlands',
         'house rykker':'houses from crownlands',
         'house slynt':'houses from crownlands',
         'house staunton':'houses from crownlands',
         'house stokeworth':'houses from crownlands',
         'house sunglass':'houses from crownlands',
         'house thorne':'houses from crownlands',
         'house velaryon':'houses from crownlands',
         'house cafferen':'houses from stormlands',#Houses from the Stormlands
         'house caron':'houses from stormlands',
         'house cole':'houses from stormlands',
         'house connington':'houses from stormlands', 
         'house dondarrion':'houses from stormlands',
         'house errol':'houses from stormlands',
         'house estermont':'houses from stormlands',
         'house fell':'houses from stormlands',
         'house gower':'houses from stormlands',
         'house grandison':'houses from stormlands',
         'house hasty':'houses from stormlands',
         'house horpe':'houses from stormlands',
         'house lonmouth':'houses from stormlands',
         'house mertyns':'houses from stormlands',
         'house morrigen':'houses from stormlands',
         'house penrose':'houses from stormlands',
         'house seaworth':'houses from stormlands',
         'house selmy':'houses from stormlands',
         'house staedmon':'houses from stormlands',
         'house swann':'houses from stormlands',
         'house tarth':'houses from stormlands',
         'house toyne':'houses from stormlands',
         'house trant':'houses from stormlands',
         'house wagstaff':'houses from stormlands',
         'house wylde':'houses from stormlands',
         'house stark':'house stark from the north',# Most important Houses
         'house nymeros martell':'house martell from dorne',
         'house martell': 'house martell from dorne',
         'house tyrell':'house tyrell from the reach',
         'house tyrell of brightwater keep':'house tyrell from the reach',
         'house lannister of casterly rock':'house Lannister of westernlands',
         'house lannister of lannisport':'house Lannister of westernlands',
         'house lannister':'house Lannister of westernlands',
         'house greyjoy':'house greyjoy from iron islands',
         'house frey':'house frey from Riverlands',
         'house frey of riverrun':'house frey from Riverlands',
         'house targaryen':'house targaryen from crownlands',
         'blacks':'house targaryen from crownlands',
         'brotherhood without banners':'rebellious group',
         'drowned men': 'house greyjoy from iron islands', 
         'band of nine': 'rebellious group',
         'brave companions':'rebellious group',
         'faceless men': 'religious group',
         'faith of the seven':'religious group',
         'graces':'religious group'})           
    
# Other houses
GoT.loc[GoT['house'].value_counts()[GoT['house']].values < 15, 'house'] = "other_house"

####################################################
# 5) Correlation Analysis
####################################################

# Building the correlation analysis
        
df_corr = GoT.corr().round(2)

print(df_corr)

df_corr.loc['isAlive'].sort_values(ascending = False)
        
# Correlation heatmap

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))
 
df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)

###############################################################################
# Factorization & Dummy Variables
###############################################################################

# Dummy Variables

# title
title_dummies = pd.get_dummies(list(GoT['title']), prefix = 'title', drop_first = True)

# culture
culture_dummies = pd.get_dummies(list(GoT['culture']), prefix = 'culture', drop_first = True)

# house
house_dummies = pd.get_dummies(list(GoT['house']), prefix = 'house', drop_first = True)

# Factorization

# Dummy in a new Dataframe
GoT_dum = pd.concat(
        [GoT.loc[:,:],
         title_dummies, culture_dummies, house_dummies],
         axis = 1)

# Deleting columns 
del(GoT_dum['title'])
del(GoT_dum['culture'])
del(GoT_dum['house'])


####################################################
#OLS (Ordinary Least Square) Method Full Model
####################################################

# Creating a statsmodel with all possible variables
# Analysing relation of various variables on isAlive (pvalue & tvalue)

lm_full = smf.ols(formula = """isAlive ~  GoT_dum['male'] +
                                          GoT_dum['book1_A_Game_Of_Thrones'] +
                                          GoT_dum['book2_A_Clash_Of_Kings'] +
                                          GoT_dum['book3_A_Storm_Of_Swords'] +
                                          GoT_dum['book4_A_Feast_For_Crows'] +
                                          GoT_dum['book5_A_Dance_with_Dragons'] +
                                          GoT_dum['isMarried'] +
                                          GoT_dum['isNoble'] +
                                          GoT_dum['age'] +
                                          GoT_dum['numDeadRelations'] +
                                          GoT_dum['popularity'] +
                                          GoT_dum['missing_title'] +
                                          GoT_dum['missing_culture'] +
                                          GoT_dum['missing_house'] +
                                          GoT_dum['missing_age'] +
                                          GoT_dum['title_maester'] +
                                          GoT_dum['title_military'] +
                                          GoT_dum['title_other_title'] +
                                          GoT_dum['title_religious-title'] +
                                          GoT_dum['title_royal'] +
                                          GoT_dum['title_unknown_title'] +
                                          GoT_dum['title_winterfell'] +
                                          GoT_dum['culture_dothraki'] +
                                          GoT_dum['culture_free-cities-across-the-narrow-sea'] +
                                          GoT_dum['culture_free_folk'] +
                                          GoT_dum['culture_ghiscari'] +
                                          GoT_dum['culture_ironborn'] +
                                          GoT_dum['culture_northmen'] +
                                          GoT_dum['culture_other_culture'] +
                                          GoT_dum['culture_reach'] +
                                          GoT_dum['culture_rivermen'] +
                                          GoT_dum['culture_unknown_culture'] +
                                          GoT_dum['culture_valemen'] +
                                          GoT_dum['culture_valyrian-peninsula'] +
                                          GoT_dum['culture_westermen'] +
                                          GoT_dum['house_house frey from Riverlands'] +
                                          GoT_dum['house_house greyjoy from iron islands'] +
                                          GoT_dum['house_house martell from dorne'] +
                                          GoT_dum['house_house stark from the north'] +
                                          GoT_dum['house_house targaryen from crownlands'] +
                                          GoT_dum['house_house tyrell from the reach'] +
                                          GoT_dum['house_houses from crownlands'] +
                                          GoT_dum['house_houses from dorne'] +                                        
                                          GoT_dum['house_houses from iron islands'] +
                                          GoT_dum['house_houses from riverlands'] +
                                          GoT_dum['house_houses from stormlands'] +
                                          GoT_dum["house_houses from the north"] +
                                          GoT_dum['house_houses from the reach'] +
                                          GoT_dum['house_houses from the vale'] +
                                          GoT_dum['house_houses from westerlands'] +
                                          GoT_dum["house_night's watch"] +
                                          GoT_dum['house_other_house'] +
                                          GoT_dum['house_rebellious group'] +
                                          GoT_dum['house_unknown_house'] -1
                                          """,
                  data = GoT_dum)


#######################
# Fitting results
results = lm_full.fit()
#######################

# RSquare value of full LM model
rsq_lm_full = results.rsquared.round(3)

# Printing summary statistics of the model
print(results.summary())


print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""") 
    
#Finding the significant model based on p-value analysis    
lm_significant= smf.ols(formula = """isAlive ~  GoT_dum['male'] +
                                          GoT_dum['book1_A_Game_Of_Thrones'] +
                                          GoT_dum['book2_A_Clash_Of_Kings'] +
                                          GoT_dum['book3_A_Storm_Of_Swords'] +
                                          GoT_dum['book4_A_Feast_For_Crows'] +
                                          GoT_dum['book5_A_Dance_with_Dragons'] +
                                          GoT_dum['isNoble'] +
                                          GoT_dum['age'] +
                                          GoT_dum['numDeadRelations'] +
                                          GoT_dum['popularity'] +
                                          GoT_dum['missing_title'] +
                                          GoT_dum['missing_house'] +
                                          GoT_dum['title_other_title'] +
                                          GoT_dum['title_religious-title'] +
                                          GoT_dum['title_royal'] +
                                          GoT_dum['title_unknown_title'] +
                                          GoT_dum['title_winterfell'] +
                                          GoT_dum['culture_dothraki'] +
                                          GoT_dum['culture_free-cities-across-the-narrow-sea'] +
                                          GoT_dum['culture_free_folk'] +
                                          GoT_dum['culture_ghiscari'] +
                                          GoT_dum['culture_ironborn'] +
                                          GoT_dum['culture_northmen'] +
                                          GoT_dum['culture_other_culture'] +
                                          GoT_dum['culture_reach'] +
                                          GoT_dum['culture_rivermen'] +
                                          GoT_dum['culture_unknown_culture'] +
                                          GoT_dum['culture_valemen'] +
                                          GoT_dum['culture_valyrian-peninsula'] +
                                          GoT_dum['culture_westermen'] +
                                          GoT_dum['house_house frey from Riverlands'] +
                                          GoT_dum['house_house greyjoy from iron islands'] +
                                          GoT_dum['house_house martell from dorne'] +
                                          GoT_dum['house_house stark from the north'] +
                                          GoT_dum['house_house targaryen from crownlands'] +
                                          GoT_dum['house_house tyrell from the reach'] +
                                          GoT_dum['house_houses from crownlands'] +
                                          GoT_dum['house_houses from dorne'] +                                        
                                          GoT_dum['house_houses from iron islands'] +
                                          GoT_dum['house_houses from riverlands'] +
                                          GoT_dum['house_houses from stormlands'] +
                                          GoT_dum["house_houses from the north"] +
                                          GoT_dum['house_houses from the reach'] +
                                          GoT_dum['house_houses from the vale'] +
                                          GoT_dum['house_houses from westerlands'] +
                                          GoT_dum["house_night's watch"] +
                                          GoT_dum['house_other_house'] +
                                          GoT_dum['house_unknown_house'] -1
                                          """,
                  data = GoT_dum)
#######################
# Fitting results
results = lm_significant.fit()
#######################

# RSquare value of full LM model
rsq_lm_significant = results.rsquared.round(3)

# Printing summary statistics of the model
print(results.summary())


print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    
####################################################
# KNN MODEL
####################################################

#Creating target variable    
GoT_target = GoT_dum.loc[:,'isAlive']

#Creating data based on the optimal model
GoT_data   = GoT_dum.loc[:,['male',
                            'book1_A_Game_Of_Thrones',
                            'book2_A_Clash_Of_Kings',
                            'book3_A_Storm_Of_Swords',
                            'book4_A_Feast_For_Crows',
                            'book5_A_Dance_with_Dragons',
                            'isNoble',
                            'age',
                            'numDeadRelations',
                            'popularity',
                            'missing_title',
                            'missing_house',
                            'title_other_title',
                            'title_religious-title',
                            'title_royal',
                            'title_unknown_title',
                            'title_winterfell',
                            'culture_dothraki',
                            'culture_free-cities-across-the-narrow-sea',
                            'culture_free_folk',
                            'culture_ghiscari',
                            'culture_ironborn',
                            'culture_northmen',
                            'culture_other_culture',
                            'culture_reach',
                            'culture_rivermen',
                            'culture_unknown_culture',
                            'culture_valemen',
                            'culture_valyrian-peninsula',
                            'culture_westermen',
                            'house_house frey from Riverlands',
                            'house_house greyjoy from iron islands',
                            'house_house martell from dorne',
                            'house_house stark from the north',
                            'house_house targaryen from crownlands',
                            'house_house tyrell from the reach',
                            'house_houses from crownlands',
                            'house_houses from dorne',
                            'house_houses from iron islands',
                            'house_houses from riverlands',
                            'house_houses from stormlands',
                            "house_houses from the north",
                            'house_houses from the reach',
                            'house_houses from the vale',
                            "house_night's watch",
                            'house_other_house',
                            'house_unknown_house']]


# Preparing test and train datsets, using the default 
X_train, X_test, y_train, y_test = train_test_split(
            GoT_data,
            GoT_target,
            test_size = 0.25,
            random_state = 508)

#Stratification 
X_train, X_test, y_train, y_test = train_test_split(
            GoT_data,
            GoT_target,
            test_size = 0.25,
            random_state = 508,
            stratify = GoT_target)

# Running the neighbor optimization with adjustment for classification
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1) 

knn_clf = KNeighborsClassifier(n_neighbors = 3)

# Fitting model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

# highest test accuracy
print(test_accuracy) 

#testing score vs training score
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

# Generating Predictions 
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test) 

########################
## Does Logistic Regression predict better than KNN?
########################
log_reg = LogisticRegression(C = 1)


log_reg_fit = log_reg.fit(X_train, y_train)

# Predictions
log_reg_pred = log_reg_fit.predict(X_test)

print(f"""
Test set predictions:
{log_reg_pred.round(2)}
""")

# Saving the prediction of results onto an excel (.xlsx) sheets
pd.DataFrame(log_reg_pred).to_excel('GOT_FLOCCO_SFMBANDD1_PredResults.xlsx', index=True)     


# Testing score vs training score.
print('Training Score', log_reg_fit.score(X_train, y_train).round(4))
print('Testing Score:', log_reg_fit.score(X_test, y_test).round(4)) 

###############################################################################
# Cross Validation with k-folds
###############################################################################

cv_knn_3 = cross_val_score(knn_clf,
                           GoT_data,
                           GoT_target,
                           cv = 3)
print(cv_knn_3)


print(pd.np.mean(cv_knn_3).round(3)) 

print('\nAverage: ',
      pd.np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))    

 

####################################################
# 10) OLS LR (Linear Regression) Significant Model
####################################################

got_OLS_train = pd.concat([X_train, y_train], axis=1)
got_OLS_test = pd.concat([X_test, y_test], axis=1)

lm_significant = smf.ols(formula = """isAlive ~   GoT_dum['male'] +
                                          GoT_dum['book1_A_Game_Of_Thrones'] +
                                          GoT_dum['book2_A_Clash_Of_Kings'] +
                                          GoT_dum['book3_A_Storm_Of_Swords'] +
                                          GoT_dum['book4_A_Feast_For_Crows'] +
                                          GoT_dum['book5_A_Dance_with_Dragons'] +
                                          GoT_dum['isNoble'] +
                                          GoT_dum['age'] +
                                          GoT_dum['numDeadRelations'] +
                                          GoT_dum['popularity'] +
                                          GoT_dum['missing_title'] +
                                          GoT_dum['missing_house'] +
                                          GoT_dum['title_other_title'] +
                                          GoT_dum['title_religious-title'] +
                                          GoT_dum['title_royal'] +
                                          GoT_dum['title_unknown_title'] +
                                          GoT_dum['title_winterfell'] +
                                          GoT_dum['culture_dothraki'] +
                                          GoT_dum['culture_free-cities-across-the-narrow-sea'] +
                                          GoT_dum['culture_free_folk'] +
                                          GoT_dum['culture_ghiscari'] +
                                          GoT_dum['culture_ironborn'] +
                                          GoT_dum['culture_northmen'] +
                                          GoT_dum['culture_other_culture'] +
                                          GoT_dum['culture_reach'] +
                                          GoT_dum['culture_rivermen'] +
                                          GoT_dum['culture_unknown_culture'] +
                                          GoT_dum['culture_valemen'] +
                                          GoT_dum['culture_valyrian-peninsula'] +
                                          GoT_dum['culture_westermen'] +
                                          GoT_dum['house_house frey from Riverlands'] +
                                          GoT_dum['house_house greyjoy from iron islands'] +
                                          GoT_dum['house_house martell from dorne'] +
                                          GoT_dum['house_house stark from the north'] +
                                          GoT_dum['house_house targaryen from crownlands'] +
                                          GoT_dum['house_house tyrell from the reach'] +
                                          GoT_dum['house_houses from crownlands'] +
                                          GoT_dum['house_houses from dorne'] +                                        
                                          GoT_dum['house_houses from iron islands'] +
                                          GoT_dum['house_houses from riverlands'] +
                                          GoT_dum['house_houses from stormlands'] +
                                          GoT_dum["house_houses from the north"] +
                                          GoT_dum['house_houses from the reach'] +
                                          GoT_dum['house_houses from the vale'] +
                                          GoT_dum['house_houses from westerlands'] +
                                          GoT_dum["house_night's watch"] +
                                          GoT_dum['house_other_house'] +
                                          GoT_dum['house_unknown_house'] -1
                                          """,
                  data = GoT_dum)


#######################
# Fitting results
results = lm_significant.fit()
results.rsquared_adj.round(3)


# Printing summary statistics of the model
print(results.summary())

rsq_lm_significant = results.rsquared.round(3)

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")










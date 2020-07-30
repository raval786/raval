# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:44:34 2020

@author: Raval
"""

import pymongo

# establish the connection with mongodb
client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

# create database
mydb = client['Employee']

# create table(Collection)
information = mydb.employeeinfo

records = {
    "Firstname":"raval"
    "Lastname":"sahil"
    "department":"AI"
    }
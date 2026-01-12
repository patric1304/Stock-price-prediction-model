               

import os

                          
HISTORY_DAYS = 30
COUNTRY_CODE = "US"

                         
 
                                            
                                                           
                                                                   
 
                                                                               
                                                                         
TARGET_MODE = os.getenv("TARGET_MODE", "logret")

                                  
                               
NEWS_DAYS_AVAILABLE = 15                                                
MAX_DAILY_REQUESTS = 100                                 
TRAINING_DATA_DAYS = 60                                      
                                                                                         
NEWS_HISTORY_DAYS = int(os.getenv("NEWS_HISTORY_DAYS", "30"))

                  
                                                                
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

                       
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

                 
INCLUDE_GLOBAL_SENTIMENT = False                                                     

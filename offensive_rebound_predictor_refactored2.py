# This program imports OKC Thunder player tracking rebounding data including player location and position data for >300,000 plays
# Files are merged and cleaned, and then various parameters are calculated such as player distance and angle from basket,
# how spread out the players are, how many offensive players are boxing out, number of players in restricted zone, etc.


# Import packages
import pandas as pd
import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from numpy import median


class Calculations:
    
    def __init__(self, loc, pbp, player_pos, player_rebound):
        
        # Import data
        self.loc_df = pd.read_csv(loc, header = 0)
        self.pbp_df = pd.read_csv(pbp, header = 0)
        self.player_pos_df = pd.read_csv(player_pos, header = 0)
        self.player_rebound_df = pd.read_csv(player_rebound)
        
    def reOrder(self):
        
        """
        Used on the training and unknown test set, this function re-orders the columns so that distance and angle
        calculations may be performed.
        """
          
        # Reorder the columns to prepare for distance from basket calculations below
        self.loc_df = self.loc_df[["game_id", "playbyplayorder_id", "row_type", "AtRim_loc_x_off_player_1", 
            "AtRim_loc_y_off_player_1",  "AtRim_loc_x_off_player_2", "AtRim_loc_y_off_player_2","AtRim_loc_x_off_player_3", 
            "AtRim_loc_y_off_player_3", "AtRim_loc_x_off_player_4", "AtRim_loc_y_off_player_4",  "AtRim_loc_x_off_player_5", 
            "AtRim_loc_y_off_player_5", "AtRim_loc_x_def_player_1", "AtRim_loc_y_def_player_1",  "AtRim_loc_x_def_player_2", 
            "AtRim_loc_y_def_player_2", "AtRim_loc_x_def_player_3", "AtRim_loc_y_def_player_3", "AtRim_loc_x_def_player_4", 
            "AtRim_loc_y_def_player_4",  "AtRim_loc_x_def_player_5", "AtRim_loc_y_def_player_5", "AtShot_loc_x_off_player_1", 
            "AtShot_loc_y_off_player_1", "AtShot_loc_x_off_player_2", "AtShot_loc_y_off_player_2", "AtShot_loc_x_off_player_3", 
            "AtShot_loc_y_off_player_3", "AtShot_loc_x_off_player_4", "AtShot_loc_y_off_player_4","AtShot_loc_x_off_player_5", 
            "AtShot_loc_y_off_player_5", "AtShot_loc_x_def_player_1", "AtShot_loc_y_def_player_1",  "AtShot_loc_x_def_player_2", 
            "AtShot_loc_y_def_player_2", "AtShot_loc_x_def_player_3", "AtShot_loc_y_def_player_3", "AtShot_loc_x_def_player_4", 
            "AtShot_loc_y_def_player_4",  "AtShot_loc_x_def_player_5", "AtShot_loc_y_def_player_5"]]
            
        
            
        return self.loc_df
        
    def clean_training(self):
        
        """
        Clean up the training data sets. A separate testing set cleaning function below is necessary because the sets have
        their own issues that must be dealt with separately.
        """
        
        # Clean up loc_df
        self.loc_df = self.loc_df.drop(self.loc_df.columns[0], axis = 1)
        # Change ' ft' to 'ft' in row_type column (i.e., remove the leading space as it prevents an accurate df merge below)
        self.loc_df = self.loc_df.replace(to_replace = " ft", value = "ft")
        
        # Clean up pbp_df
        self.pbp_df = self.pbp_df.drop(self.pbp_df.columns[0], axis = 1)
        self.pbp_df = self.pbp_df.drop(self.pbp_df.loc[:, 'reboffensive': 'eventdescription'].columns, axis = 1)
        # Change 'final ft' to ft in row_type column 
        self.pbp_df = self.pbp_df.replace(to_replace = "final ft", value = "ft")
        
        # Clean up position data
        self.player_pos_df = self.player_pos_df.drop(self.player_pos_df.columns[[0, 3]], axis = 1)
        
        # Clean up rebounding data
        self.player_rebound_df = self.player_rebound_df.drop(self.player_rebound_df.columns[0], axis = 1)
        
        return self.loc_df, self.pbp_df, self.player_pos_df, self.player_rebound_df
        
    def clean_testing(self):
        
        # Clean up (this isn't needed for the unknown test data
        # self.loc_df = self.loc_df.drop(self.loc_df.columns[0], axis = 1)
        
        # Clean up
        self.pbp_df = self.pbp_df.drop(self.pbp_df.loc[:, 'off_team_id': 'eventdescription'].columns, axis = 1)
        
        # Clean  up
        self.player_pos_df = self.player_pos_df.drop(self.player_pos_df.columns[[0, 3]], axis = 1)
        
        # Clean up
        self.player_rebound_df = self.player_rebound_df.drop(self.player_rebound_df.columns[0], axis = 1)
        
 
        
        return self.loc_df, self.pbp_df, self.player_pos_df, self.player_rebound_df
        
        
    def merge(self):
        
        """
        Merge the location and pbp data sets.
        """
        
        # Merge self.loc_df and self.pbp_df on playbyplayorder_id
        self.pbp_loc_merged_df = pd.merge(self.pbp_df, self.loc_df, on = ["playbyplayorder_id", "row_type", "game_id"])
        
        # Make changes in actiondescription columns to simplify
        self.pbp_loc_merged_df["actiondescription"] = self.pbp_loc_merged_df["actiondescription"].apply(lambda x: 'jump_shot' if 'Jump' in x else ('layup' if 'Layup' in x else ('other')))   
        
        # Inspect the data
        """
        print (self.pbp_loc_merged_df.info())
        self.pbp_loc_merged_df.hist(bins = 50, figsize = (20, 15))
        plt.show()
        """
        
        
        return self.pbp_loc_merged_df
        
    def purge_oreb(self):
        
        """
        Remove all rows in the training set that have NA in the oreb column.
        This can only be called on the training data since the testing data doesn't have oreb.
        """
        
        # Remove rows that have NA in the oreb column. Presumably these are team rebounds, out of bounds, fouls, makes, etc.
        self.pbp_loc_merged_df = self.pbp_loc_merged_df.dropna(subset = ["oreb"]).reset_index(drop = True)
        
        # Remove rows in which the shot was a free throw as these are almost never rebounded by the offense and player positions are fixed
        # 174,141 rows remaining
        self.pbp_loc_merged_df = self.pbp_loc_merged_df[~self.pbp_loc_merged_df.row_type.str.contains('ft').reset_index(drop = True)]
        
        return self.pbp_loc_merged_df
    
    def purge_shotLoc(self):
        
        """
        Remove all rows with NA in a location row.
        This can be used on both the training and testing sets.
        """
        
        # Remove rows with NA in a location row
        # df is now 173744 rows long; still plenty of data
        self.pbp_loc_merged_df = self.pbp_loc_merged_df.dropna(subset = ["AtRim_loc_x_off_player_1", 
            "AtRim_loc_y_off_player_1",  "AtRim_loc_x_off_player_2", "AtRim_loc_y_off_player_2","AtRim_loc_x_off_player_3", 
            "AtRim_loc_y_off_player_3", "AtRim_loc_x_off_player_4", "AtRim_loc_y_off_player_4",  "AtRim_loc_x_off_player_5", 
            "AtRim_loc_y_off_player_5", "AtRim_loc_x_def_player_1", "AtRim_loc_y_def_player_1",  "AtRim_loc_x_def_player_2", 
            "AtRim_loc_y_def_player_2", "AtRim_loc_x_def_player_3", "AtRim_loc_y_def_player_3", "AtRim_loc_x_def_player_4", 
            "AtRim_loc_y_def_player_4",  "AtRim_loc_x_def_player_5", "AtRim_loc_y_def_player_5", "AtShot_loc_x_off_player_1", 
            "AtShot_loc_y_off_player_1", "AtShot_loc_x_off_player_2", "AtShot_loc_y_off_player_2", "AtShot_loc_x_off_player_3", 
            "AtShot_loc_y_off_player_3", "AtShot_loc_x_off_player_4", "AtShot_loc_y_off_player_4","AtShot_loc_x_off_player_5", 
            "AtShot_loc_y_off_player_5", "AtShot_loc_x_def_player_1", "AtShot_loc_y_def_player_1",  "AtShot_loc_x_def_player_2", 
            "AtShot_loc_y_def_player_2", "AtShot_loc_x_def_player_3", "AtShot_loc_y_def_player_3", "AtShot_loc_x_def_player_4", 
            "AtShot_loc_y_def_player_4",  "AtShot_loc_x_def_player_5", "AtShot_loc_y_def_player_5"])
            
        
                
        return self.pbp_loc_merged_df
            
        
        
    def distance_from_hoop(self):
        
        """
        Calculate distance of every player from basket at time of shot and when ball hits the rim.
        Use on both training and testing sets.
        """
        
        # Copy self.pbp_loc_merged_df
        self.pbp_loc_player_distance_df = self.pbp_loc_merged_df.copy()
        
        # One of the locations in the unknown test set is an object instead of float for some reason-convert
        self.pbp_loc_player_distance_df['AtShot_loc_x_off_player_2'] = self.pbp_loc_player_distance_df['AtShot_loc_x_off_player_2'].astype(float)
        
        # Calculate distance from basket for every x-y pair (each player) using the pythagorean theorom. 
        # Column order matters here! Must go AtRim_loc_x_off_player1, AtRim_loc_y_off_player1, AtRim_loc_x_off_player2...
        # Must be a block of AtRim followed by a block of AtShot
        # Get column index of first x and y positions
        self.x_col = self.pbp_loc_player_distance_df.columns.get_loc("AtRim_loc_x_off_player_1")
        self.y_col = self.pbp_loc_player_distance_df.columns.get_loc("AtRim_loc_y_off_player_1") 
        
        # Create counter for keeping track of column positions and naming new columns
        self.x_index = 1
        
        # While loop to move through x-y pairs until all distances have been calculated
        # Note the last x column is 39 cols past the first; the last y column is 40 cols past the first (10 off + 10 def)* 2 coords each
        while self.x_index <= 39:
            # If statement to keep track of player numbers based upon self.x_index
            # Calculate player number
            self.player = 0.5*(self.x_index + 1)
            
            # If statement so that when self.x_index > 5, self.player is reset to 1 for the defensive players
            if self.player <= 5:
                self.player = 'AtRim_distance_off_player_' + str(self.player)
                
            elif 5 < self.player <= 10:
                self.player = 'AtRim_distance_def_player_' + str(self.player - 5)
                
            elif 10 < self.player <= 15:
                self.player = 'AtShot_distance_off_player_' + str(self.player - 10)
                
            else:
                self.player = 'AtShot_distance_def_player_' + str(self.player - 15)
            
            # Add the calculated distance for each x-y pair to the df as a new column
            self.pbp_loc_player_distance_df[self.player] = self.pbp_loc_player_distance_df.apply(lambda row: self.pythagorean
                                                                (row[self.x_col], row[self.y_col]), axis = 1)
                   
            # Increment x and y index values and columns
            self.x_index += 2
            # self.y_index += 2
            self.x_col += 2
            self.y_col += 2     
        
        # Inspect the data
        # print (self.pbp_loc_player_distance_df.info())
        
        # self.pbp_loc_player_distance_df.hist(bins = 50, figsize = (20, 15))
        # plt.show()
        
        
        
        
        return self.pbp_loc_player_distance_df
        
        
#----------------------------------------------------------------------------------------------   
    # Function to use with apply to caculate player distance from basket
    # Include 41.75 in the formula to reset the basket as the origin in the x-direction
    def pythagorean(self, a, b):
        
        return math.sqrt((a + 41.75)**2 + b**2)
        
#----------------------------------------------------------------------------------------------            
        

    def distance_vc_from_basket(self):
        
        """
        Function to calculate the variation coefficient for the distance of every player from the basket
        Variation coefficient = (STDEV of distance from basket for offensive or defensive players)/mean distance from basket of offense or defense
        Hypothesis: The ability to get a rebound depends upon the variation in distance of players from the basket
        """
        
        # Get first column index 
        self.first_col = self.pbp_loc_player_distance_df.columns.get_loc("AtRim_distance_off_player_1.0")
        self.second_col = self.first_col + 1
        self.third_col = self.first_col + 2
        self.fourth_col = self.first_col + 3
        self.fifth_col = self.first_col + 4
        
        self.x_cntr = 1
        
        while self.x_cntr <= 4:
            
            if self.x_cntr == 1:
                self.header = 'AtRim_off_dis_vc'
               
            elif self.x_cntr == 2:
                self.header = 'AtRim_def_dis_vc' 
                
            elif self.x_cntr == 3:
                self.header = 'AtShot_off_dis_vc' 
                
            else:
                self.header = 'AtShot_def_dis_vc'
            
            self.pbp_loc_player_distance_df[self.header] = self.pbp_loc_player_distance_df.apply(lambda row: self.vc_calculator
                (row[self.first_col], row[self.second_col], row[self.third_col], row[self.fourth_col], row[self.fifth_col]), axis = 1)
               
            self.first_col += 5
            self.second_col += 5
            self.third_col += 5
            self.fourth_col += 5
            self.fifth_col  += 5
            self.x_cntr += 1
      
      
        return self.pbp_loc_player_distance_df
           
    #----------------------------------------------------------------------------------------------    
    # Function for use with apply in the calculation of player distance variation coeficient (vc)
    
    def vc_calculator(self, a, b, c, d, e):
        
        number_list = [a, b, c, d, e]
       
        mean = sum(number_list)/len(number_list)
        variance = sum([((x - mean)**2) for x in number_list])/(len(number_list)-1) 
        stdev = variance**0.5
            
        vc = stdev/mean  
        
       
        
        return vc
        
    #----------------------------------------------------------------------------------------------        
        
    # Function to include whether the top offensive and defensive rebounders are in the game and how many are in the game
    def top_rebounders(self):
        
        # Create list of top rebounders
        self.top_rebounders_list = [83, 275, 298, 81, 22, 108, 120, 170, 13, 26, 127, 823, 145, 717, 889, 341]
        
        # Set all NAs to '-' in the playerID columns
        self.pbp_loc_player_distance_df = self.pbp_loc_player_distance_df.fillna({'playerid_off_player_1': '-', 'playerid_off_player_2': '-', 
            'playerid_off_player_3': '-', 'playerid_off_player_4': '-', 'playerid_off_player_5': '-'})
        
        
        # Create a master list of lists (This will be converted to a df and merged with self.pbp_loc_player_distance_restricted_zone_df
        self.master_playa_dummy_list = []
        
        # Also get a count of the number of top rebounders in each play 
        # Create a list to hold the count
        self.no_top_rebounders_list = []

        for row in self.pbp_loc_player_distance_df.itertuples():
            
            # Create list of playerIDs
            self.playerID_list = [row.playerid_off_player_1, row.playerid_off_player_2, row.playerid_off_player_3, row.playerid_off_player_4, row.playerid_off_player_5]
            
            # Create temp list of 0s and 1s for each player. This will be appended to self.master_playa_dummy_list to create a list of lists.
            self.playa_dummy_list = []
            
            # Start a counter for the number of top rebounders in the play
            self.no_rebounders_inPlay = 0
            
            # Compare self.playerID_list and self.top_rebounders_list for matches
            for rebounder in self.top_rebounders_list:
                
                if rebounder in self.playerID_list:
                    self.playa_dummy_list.append(1)
                    self.no_rebounders_inPlay += 1
                    
                else:
                    self.playa_dummy_list.append(0)
            
            # Append to master list to create a list of lists for each player
            self.master_playa_dummy_list.append(self.playa_dummy_list)
            
            # Append number of rebounders count to master list
            self.no_top_rebounders_list.append(self.no_rebounders_inPlay) 
            
        # Create dataframe from list of lists
        self.playa_df = pd.DataFrame(self.master_playa_dummy_list, columns = ['player83', 'player275', 'player298', 'player81', 'player22', 'player108', 'player120', 'player170', 'player13', 'player26',
                            'player127', 'player823', 'player145', 'player717', 'player889', 'player341'])
        
        # Reset index values on both of the dataframes before concatenating or blank rows will be inserted
        self.playa_df = self.playa_df.reset_index(drop = True)
        self.pbp_loc_player_distance_df = self.pbp_loc_player_distance_df.reset_index(drop = True)
        
        # Concatenate self.playa_df and self.pbp_loc_player_distance_restricted_zone_df
        self.pbp_loc_player_distance_df = pd.concat([self.pbp_loc_player_distance_df, self.playa_df], axis = 1)
        
        # Append self.no_top_rebounders_list to df
        self.pbp_loc_player_distance_df['no_of_top_rebounders_in_play'] = self.no_top_rebounders_list
        
     
        return self.pbp_loc_player_distance_df
        
        
    def build_training_set(self):
        
        # Convert Yes to 1 and No to 0 in oreb column
        self.pbp_loc_player_distance_df['oreb'] = self.pbp_loc_player_distance_df['oreb'].str.replace('Yes','1')
        self.pbp_loc_player_distance_df['oreb'] = self.pbp_loc_player_distance_df['oreb'].str.replace('No','0')
        
        # Change oreb from string to int
        self.pbp_loc_player_distance_df['oreb'] = self.pbp_loc_player_distance_df['oreb'].astype(int) 
        
        # One hot encode actiondescription
        self.pbp_loc_player_distance_df = pd.get_dummies(self.pbp_loc_player_distance_df, columns=["actiondescription"])
        
        # Train 
        self.X = self.pbp_loc_player_distance_df[['no_of_top_rebounders_in_play','actiondescription_jump_shot', 'actiondescription_layup', 'AtRim_off_dis_vc', 'AtRim_def_dis_vc']]       
        self.y = self.pbp_loc_player_distance_df['oreb']
        
        return self.X, self.y
        
    def split_train_test(self):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.5, random_state = 1)
        
        self.model = LogisticRegression()
        self.modelFit = self.model.fit(self.X_train, self.y_train)
        self.probs = self.model.predict_proba(self.X_test)[:, 1]
        
        
        false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(self.y_test, self.probs)
        print('roc_auc_score for Logistic Regression: ', roc_auc_score(self.y_test, self.probs))
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.probs)
        plt.plot([0, 1], [0, 1], linestyle = '--')
        # Plot the roc curve for the model
        plt.plot(fpr, tpr)
        # Show the plot
        plt.show()
        
        
        # Calculate logloss
        self.log_loss = log_loss(self.y_test, self.probs)
        print('log loss: ', self.log_loss)
        
        
        # Process the unknown data files
        # Enter unknown data files in the order location, pbp, player position, and player rebound
        obj2 = Calculations(path + 'okc_testing_data_loc_dev_set.csv', path + 'okc_testing_data_pbp_dev_set.csv', path + 'player_pos_data.csv', path + 'player_reb_data.csv')
        
        # Re-order the columns
        self.reOrderUnk = obj2.reOrder()
        # Clean up and merge the data, and run calculations
        self.clean_up_Unk = obj2.clean_testing()
        self.mergeUnk = obj2.merge()
        self.purgeShotLocUnk = obj2.purge_shotLoc()
        self.distanceUnk = obj2.distance_from_hoop()
        self.varitionUnk = obj2.distance_vc_from_basket()
        self.reboundersUnk = obj2.top_rebounders()
        
        # One hot encode actiondescription in the testUnk_df
        self.reboundersUnk = pd.get_dummies(self.reboundersUnk, columns = ["actiondescription"])
        
        # Create the test set
        self.X_test = self.reboundersUnk[['no_of_top_rebounders_in_play','actiondescription_jump_shot', 'actiondescription_layup', 'AtRim_off_dis_vc', 'AtRim_def_dis_vc']]
        
        # Predict the unknown data using the model and self.X_test
        self.probs = self.model.predict_proba(self.X_test)[:, 1]
        
        # Convert to df for export with playbyplayorder_id as one of the columns
        self.play_id = list(self.reboundersUnk['playbyplayorder_id'])
        self.probs_df = pd.DataFrame()
        self.probs_df['playbyplayorder_id'] = self.play_id
        self.probs_df['probability'] = self.probs
        
        
        # Export probability predictions to csv
        self.probs_df.to_csv("/Users/danfeldheim/Documents/unknown_prediction_test1.csv", index = False)
         
# Call calculations class to process the training data     
# Enter path to csv files
path = "/Users/danfeldheim/Documents/OKC_project/"

# Enter training files in obj1 in the order location, pbp, player position, and player rebound
# Note unknown files must be entered on line 414.
obj1 = Calculations(path + '10_22_20_loc_test_set.csv', path + '10_22_20_pbp_test_set.csv', path + 'player_pos_data.csv', path + 'player_reb_data.csv')
# Any of the df objects in init can be accessed
# print (obj1.loc_df)

# Re-order the columns
reOrder = obj1.reOrder

# Clean up the data
clean_up_training = obj1.clean_training()
# How to access each returned object separately
# print (clean_up_training[0])

merge = obj1.merge()
purgeOreb = obj1.purge_oreb()
purgeShotLoc = obj1.purge_shotLoc()
distance = obj1.distance_from_hoop()
varition = obj1.distance_vc_from_basket()
rebounders = obj1.top_rebounders()
trainSet = obj1.build_training_set()
train_test = obj1.split_train_test()








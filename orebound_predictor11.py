# This program imports OKC Thunder player tracking rebounding data including player location and position data for >300,000 plays
# Files are merged and cleaned, and then various parameters are calculated such as player distance and angle from basket,
# how spread out the players are, how many offensive players are boxing out, number of players in restricted zone, etc.
# Import file names are entered in the import_files function
# This program has not been refactored! For a refactored version, see offensive_rebound_predictor_refactored2.py

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
from sklearn.model_selection import cross_val_score
from numpy import median


class rebound:
    
    def __init__(self):
        pass
        
    def import_files(self):
        
        # Import data
        # Location data
        # Full set
        # OKC location data training set
        self.loc_df = pd.read_csv("/Users/danfeldheim/Documents/OKC_project/training_data_loc.csv", header = 0)
        
        # Small location development set
        # self.loc_df = pd.read_csv("/Users/danfeldheim/Documents/OKC_project/10_22_20_loc_test_set.csv", header = 0)
        
        # Clean up
        self.loc_df = self.loc_df.drop(self.loc_df.columns[0], axis = 1)
        # Change ' ft' to 'ft' in row_type column (i.e., remove the leading space as it prevents an accurate df merge below)
        self.loc_df = self.loc_df.replace(to_replace = " ft", value = "ft")
       
        # pbp data
        # Full set
        # OKC pbp training set
        self.pbp_df = pd.read_csv("/Users/danfeldheim/Documents/OKC_project/training_data_pbp.csv", header = 0)
        
        # Small pbp development set
        # self.pbp_df = pd.read_csv("/Users/danfeldheim/Documents/OKC_project/10_22_20_pbp_test_set.csv", header = 0)
        
        # Clean up
        self.pbp_df = self.pbp_df.drop(self.pbp_df.columns[0], axis = 1)
        self.pbp_df = self.pbp_df.drop(self.pbp_df.loc[:, 'reboffensive': 'eventdescription'].columns, axis = 1)
        
        # Change 'final ft' to ft in row_type column 
        self.pbp_df = self.pbp_df.replace(to_replace = "final ft", value = "ft")
        
        # Position data
        self.player_pos_data_df = pd.read_csv("/Users/danfeldheim/Documents/OKC_project/player_pos_data.csv", header = 0)
        # Clean  up
        self.player_pos_data_df = self.player_pos_data_df.drop(self.player_pos_data_df.columns[[0, 3]], axis = 1)
        
        # Rebounding data
        self.player_reb_data_df = pd.read_csv("/Users/danfeldheim/Documents/OKC_project/player_reb_data.csv", header = 0)
        # Clean up
        self.player_reb_data_df = self.player_reb_data_df.drop(self.player_reb_data_df.columns[0], axis = 1)
        
        return self.loc_df, self.pbp_df, self.player_pos_data_df, self.player_reb_data_df
        
        
    def merge(self):
        
        # Merge self.loc_df and self.pbp_df on playbyplayorder_id
        self.pbp_loc_merged_df = pd.merge(self.loc_df, self.pbp_df, on = ["playbyplayorder_id", "row_type", "game_id"])
        
        """
        # Inspect the data
        print (a.info())
        self.pbp_loc_merged_df.hist(bins = 50, figsize = (20, 15))
        plt.show()
        """
        
        return self.pbp_loc_merged_df
        
         
    def prep(self):

        # Remove rows that have NA in the oreb column. Presumably these are team rebounds, out of bounds, fouls, makes, etc.
        self.pbp_loc_merged_df = self.pbp_loc_merged_df.dropna(subset = ["oreb"]).reset_index(drop = True)
        
        # Remove rows in which the shot was a free throw as these are almost never rebounded by the offense and player positions are fixed
        # 174,141 rows remaining
        self.pbp_loc_merged_df = self.pbp_loc_merged_df[~self.pbp_loc_merged_df.row_type.str.contains('ft').reset_index(drop = True)]
        
        # Convert Yes to 1 and No to 0 in oreb column
        self.pbp_loc_merged_df['oreb'] = self.pbp_loc_merged_df['oreb'].str.replace('Yes','1')
        self.pbp_loc_merged_df['oreb'] = self.pbp_loc_merged_df['oreb'].str.replace('No','0')
        
        # Change oreb from string to int
        self.pbp_loc_merged_df['oreb'] = self.pbp_loc_merged_df['oreb'].astype(int) 
        
        # Make changes in actiondescription columns
        # If including free throws
        # self.pbp_df["actiondescription"] = self.pbp_df["actiondescription"].apply(lambda x: 'jump_shot' if 'Jump' in x else ('layup' if 'Layup' in x else ('free_throw' if 'Free' in x else ('other'))))
        # No free throws
        self.pbp_loc_merged_df["actiondescription"] = self.pbp_loc_merged_df["actiondescription"].apply(lambda x: 'jump_shot' if 'Jump' in x else ('layup' if 'Layup' in x else ('other')))   
        
        # Convert these to categories
        self.pbp_loc_merged_df["actiondescription"] = self.pbp_loc_merged_df["actiondescription"].astype('category')
        # Convert categories to dummy numbers (adds a new column with codes: other = 2, layup = 1, jumpshot = 0)
        self.pbp_loc_merged_df["actiondescription_dummy"] = self.pbp_loc_merged_df["actiondescription"].cat.codes
        
        
        return self.pbp_loc_merged_df
    
    # Function to add player position data to the df if desired   
    def player_pos_data(self):
        
        # Add player position data from self.player_pos_data_df
        # Start a counter to keep track of player column
        self.player_column = 1
        
        # Get column number for playerid_off_player_1 for use as a counter to move through player columns
        self.column_counter = self.pbp_loc_merged_df.columns.get_loc("playerid_off_player_1")
        # Adding 1 for use with itertuples, which counts column index values differently for some reason
        self.column_counter = self.column_counter + 1 
        
         # While loop to move through player columns until all 10 player positions have been found
        while self.player_column <= 10:
            # Use counter to create the appropriate column names for the positions of all 10 players
            # If statement so that when self.player_column > 5, self.player_column is reset to 1 for the defensive players
            # Same when transitioning from offensive to defensive players
            if self.player_column <= 5:
                self.player = 'player_pos_off_player_' + str(self.player_column)
                
            else:
                self.player = 'player_pos_def_player_' + str(self.player_column)
                
            # Create a list for all of the player positions for all players in the current row
            # This list will be appended to self.pbp_loc_merged_df
            self.position_list = []
            
            # Loop through player_id rows
            for row in self.pbp_loc_merged_df.itertuples():
                
                self.player_id = row[self.column_counter]
                
                # Search id column of self.player_pos_data_df for match and return row 
                self.player_row = self.player_pos_data_df[self.player_pos_data_df['player_id'] == self.player_id]
                          
                # Retreive player position
                self.player_pos = int(self.player_row.iloc[0]['position'])
                
                # Append self.player_pos to self.position_list
                self.position_list.append(self.player_pos)
                
            # Append self.position_list to self.pbp_loc_merged_df
            self.pbp_loc_merged_df[self.player] = self.position_list
               
            # Increment column counters
            self.player_column += 1
            self.column_counter += 1
            

        return self.pbp_loc_merged_df
     
        
    def distance_from_hoop(self):
        
        # Calculate the distance of every player from the basket when the shot was taken and when it hit the rim
        # Create a new df 
        # Add parameters to this df as desired
        self.pbp_loc_player_distance_df = self.pbp_loc_merged_df[["playbyplayorder_id", 
            "AtRim_loc_x_off_player_1", "AtRim_loc_y_off_player_1",  "AtRim_loc_x_off_player_2", "AtRim_loc_y_off_player_2",
            "AtRim_loc_x_off_player_3", "AtRim_loc_y_off_player_3", "AtRim_loc_x_off_player_4", "AtRim_loc_y_off_player_4",  
            "AtRim_loc_x_off_player_5", "AtRim_loc_y_off_player_5", "AtRim_loc_x_def_player_1", "AtRim_loc_y_def_player_1",  
            "AtRim_loc_x_def_player_2", "AtRim_loc_y_def_player_2", "AtRim_loc_x_def_player_3", "AtRim_loc_y_def_player_3", 
            "AtRim_loc_x_def_player_4", "AtRim_loc_y_def_player_4",  "AtRim_loc_x_def_player_5", "AtRim_loc_y_def_player_5",
            "AtShot_loc_x_off_player_1", "AtShot_loc_y_off_player_1", "AtShot_loc_x_off_player_2", "AtShot_loc_y_off_player_2", 
            "AtShot_loc_x_off_player_3", "AtShot_loc_y_off_player_3", "AtShot_loc_x_off_player_4", "AtShot_loc_y_off_player_4",
            "AtShot_loc_x_off_player_5", "AtShot_loc_y_off_player_5", "AtShot_loc_x_def_player_1", "AtShot_loc_y_def_player_1",  
            "AtShot_loc_x_def_player_2", "AtShot_loc_y_def_player_2", "AtShot_loc_x_def_player_3", "AtShot_loc_y_def_player_3",  
            "AtShot_loc_x_def_player_4", "AtShot_loc_y_def_player_4",  "AtShot_loc_x_def_player_5", "AtShot_loc_y_def_player_5", "oreb", 
            'playerid_off_player_1', 'playerid_off_player_2', 'playerid_off_player_3', 'playerid_off_player_4', 'playerid_off_player_5',
            'playerid_def_player_1', 'playerid_def_player_2','playerid_def_player_3','playerid_def_player_4','playerid_def_player_5', 
            'actiondescription_dummy']]
            
        # Remove rows with NA in a location row
        # df is now 173744 rows long; still plenty of data
        self.pbp_loc_player_distance_df = self.pbp_loc_player_distance_df.dropna(subset = ["AtRim_loc_x_off_player_1", 
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
        
        
        # Calculate distance from basket for every x-y pair (each player) using the pythagorean theorom. 
        # Get column index of first x and y positions
        self.x_col = self.pbp_loc_player_distance_df.columns.get_loc("AtRim_loc_x_off_player_1")
        self.y_col = self.pbp_loc_player_distance_df.columns.get_loc("AtRim_loc_y_off_player_1") 
        
        # Create counter for keeping track of column positions and naming new columns
        # Could fix this with a try/except, but its unlikely that the first data col will be 0
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
            
        
        """
        # Inspect the data
        # print (self.pbp_loc_player_distance_df.info())
        self.pbp_loc_player_distance_df.hist(bins = 50, figsize = (20, 15))
        plt.show()
        """
        
        return self.pbp_loc_player_distance_df
      
        
    
    #----------------------------------------------------------------------------------------------   
    # Function to use with apply to caculate player distance from basket
    # Include 41.75 in the formula to reset the basket as the origin in the x-direction
    def pythagorean(self, a, b):
        
        return math.sqrt((a + 41.75)**2 + b**2)
        
    #----------------------------------------------------------------------------------------------
    
    
    # Functions to calculate the number of offensive and defensive players within x feet of rim (started with x = 6)
    def restricted_players(self):
        
        # Copy self.pbp_loc_rim_distance_df
        self.pbp_loc_player_distance_restricted_zone_df = self.pbp_loc_player_distance_df.copy()
        
        # Add the calculated number of offensive players in restricted zone when ball hits rim as new column
        self.pbp_loc_player_distance_restricted_zone_df['AtRim_restricted_off_players'] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.AtRim_off_players_in_restricted_zone
            (row['AtRim_distance_off_player_1.0'], row['AtRim_distance_off_player_2.0'], row['AtRim_distance_off_player_3.0'], row['AtRim_distance_off_player_4.0'], row['AtRim_distance_off_player_5.0']), axis = 1)
        
        # Add the calculated number of defensive players in restricted zone when ball hits rim as new column
        self.pbp_loc_player_distance_restricted_zone_df['AtRim_restricted_def_players'] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.AtRim_def_players_in_restricted_zone
            (row['AtRim_distance_def_player_1.0'], row['AtRim_distance_def_player_2.0'], row['AtRim_distance_def_player_3.0'], row['AtRim_distance_def_player_4.0'], row['AtRim_distance_def_player_5.0']), axis = 1)
            
        # Add the calculated number of offensive players in restricted zone when shot is taken as new column    
        self.pbp_loc_player_distance_restricted_zone_df['AtShot_restricted_off_players'] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.AtShot_off_players_in_restricted_zone
            (row['AtShot_distance_off_player_1.0'], row['AtShot_distance_off_player_2.0'], row['AtShot_distance_off_player_3.0'], row['AtShot_distance_off_player_4.0'], row['AtShot_distance_off_player_5.0']), axis = 1)   
            
        # Add the calculated number of defensive players in restricted zone when shot is taken as new column 
        self.pbp_loc_player_distance_restricted_zone_df['AtShot_restricted_def_players'] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.AtShot_def_players_in_restricted_zone(   
            row['AtShot_distance_def_player_1.0'], row['AtShot_distance_def_player_2.0'], row['AtShot_distance_def_player_3.0'], row['AtShot_distance_def_player_4.0'], row['AtShot_distance_def_player_5.0']), axis = 1)   
            
        
        # Tested calculations and confirmed
        # self.pbp_loc_player_distance_restricted_zone_df.to_csv('/Users/danfeldheim/Documents/OKC_project/restricted_zone_test_df.csv') 
        
        return self.pbp_loc_player_distance_restricted_zone_df
        
       
        
    #----------------------------------------------------------------------------------------------
    # Functions for use with apply in the calculation of number of players in restricted zone
        
    # Do calculation and return number of offensive and defensive players within x feet of rim
    # a-e represent offensive players; j-n represent defensive players
    def AtRim_off_players_in_restricted_zone(self, a, b, c, d, e):
        
        # Create a list of offensive and defensive player distances
        self.AtRim_off_player_dist_list = [a, b, c, d, e]
        
        # Create a counter for number of offensive defensive players in restricted zone
        self.AtRim_off = 0
        
        for entry in self.AtRim_off_player_dist_list:
            if entry < 6:
                self.AtRim_off += 1

        
        return float(self.AtRim_off)
        
    def AtRim_def_players_in_restricted_zone(self, j, k, l, m, n):
       
         # Create a list of offensive and defensive player distances
        self.AtRim_def_player_dist_list = [j, k, l, m, n]
        
        # Create a counter for number of offensive defensive players in restricted zone
        self.AtRim_def = 0
        
        for entry in self.AtRim_def_player_dist_list:
            if entry < 6:
                self.AtRim_def += 1
                
        return float(self.AtRim_def)
        
        
    def AtShot_off_players_in_restricted_zone(self, a, b, c, d, e):
        
        # Create a list of offensive and defensive player distances
        self.AtShot_off_player_dist_list = [a, b, c, d, e]
        
        # Create a counter for number of offensive defensive players in restricted zone
        self.AtShot_off = 0
        
        for entry in self.AtShot_off_player_dist_list:
            if entry < 6:
                self.AtShot_off += 1

        
        return float(self.AtShot_off)
        
    def AtShot_def_players_in_restricted_zone(self, j, k, l, m, n):
       
         # Create a list of offensive and defensive player distances
        self.AtShot_def_player_dist_list = [j, k, l, m, n]
        
        # Create a counter for number of offensive defensive players in restricted zone
        self.AtShot_def = 0
        
        for entry in self.AtShot_def_player_dist_list:
            if entry < 6:
                self.AtShot_def += 1
                
        return float(self.AtShot_def)
        
    #----------------------------------------------------------------------------------------------    
        
    # Calculate player angles
    def player_angles(self):
        
        # Calculate angle in degrees from basket for every x-y pair (each player) using the tangent formula: tan(theta) = Opposite/Adjacent = y coord/x coord. 
        # Create counters to toggle through the x-y pairs (start at an index of 1 for AtRim_loc_x_off_player_1 and 2 for AtRim_loc_y_off_player_1)
        # Calculate distance from basket for every x-y pair (each player) using the pythagorean theorom. 
        # Get column index of first x and y positions
        self.x_col = self.pbp_loc_player_distance_restricted_zone_df.columns.get_loc("AtRim_loc_x_off_player_1")
        self.y_col = self.pbp_loc_player_distance_restricted_zone_df.columns.get_loc("AtRim_loc_y_off_player_1")
        
        # Set x index to 1 for use as counter and new column names
        self.x_index = 1
        
        # While loop to move through x-y pairs until all angles have been calculated (note last x col is +39 from first x col)
        while self.x_index <= 39:
            # If statement to keep track of player numbers based upon self.x_index
            # Calculate player number
            self.player = 0.5*(self.x_index + 1)
            
            # If statement so that when self.x_index > 5, self.player is reset to 1 for the defensive players
            # Same when transitioning from offensive to defensive players
            if self.player <= 5:
                self.player = 'AtRim_angle_off_player_' + str(self.player)
                
            elif 5 < self.player <= 10:
                self.player = 'AtRim_angle_def_player_' + str(self.player - 5)
                
            elif 10 < self.player <= 15:
                self.player = 'AtShot_angle_off_player_' + str(self.player - 10)
                
            else:
                self.player = 'AtShot_angle_def_player_' + str(self.player - 15)
            
                
            self.pbp_loc_player_distance_restricted_zone_df[self.player] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.angle_calculator
                (row[self.x_col], row[self.y_col]), axis = 1)
                
            # Increment x and y index values
            self.x_index += 2
            self.x_col += 2
            self.y_col += 2
            
        # Tested and confirmed
        # self.pbp_loc_player_distance_restricted_zone_df.to_csv('/Users/danfeldheim/Documents/OKC_project/angle_test_df2.csv') 
     
        return self.pbp_loc_player_distance_restricted_zone_df
        
      
     
    
    #----------------------------------------------------------------------------------------------    
    # Function for use with apply in the calculation of player angles
    
    def angle_calculator(self, a, b):
        
        return math.degrees(math.atan(b/a))
        
    #----------------------------------------------------------------------------------------------                          
                    
    # Function to calculate the variation coefficient for the distance of every player from the basket
    # Variation coefficient = (STDEV of distance from basket for offensive or defensive players)/mean distance from basket of offense or defense
    # Hypothesis: The ability to get a rebound depends upon the variation in distance of players from the basket
    # The same calculation will be done for player's distance from other players
    def distance_vc_from_basket(self):
        
        # Get first column index 
        self.first_col = self.pbp_loc_player_distance_restricted_zone_df.columns.get_loc("AtRim_distance_off_player_1.0")
        self.second_col = self.first_col + 1
        self.third_col = self.first_col + 2
        self.fourth_col = self.first_col + 3
        self.fifth_col = self.first_col + 4
        
        self.x_cntr = 1
        
        while self.x_cntr <= 4:
            
            if self.x_cntr == 1:
                self.header = 'AtRim_off_dis_vc'
               
            # elif self.x_cntr == self.x_cntr + 5:
            elif self.x_cntr == 2:
                self.header = 'AtRim_def_dis_vc' 
                
            # elif self.x_cntr + 10 <= self.x_cntr <= self.cntr + 15:
            elif self.x_cntr == 3:
                self.header = 'AtShot_off_dis_vc' 
                
            else:
                self.header = 'AtShot_def_dis_vc'
            
            self.pbp_loc_player_distance_restricted_zone_df[self.header] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.vc_calculator
                (row[self.first_col], row[self.second_col], row[self.third_col], row[self.fourth_col], row[self.fifth_col]), axis = 1)
             
            
            self.first_col += 5
            self.second_col += 5
            self.third_col += 5
            self.fourth_col += 5
            self.fifth_col  += 5
            self.x_cntr += 1
            
            
        # Tested and calculations confirmed 
        # self.pbp_loc_player_distance_restricted_zone_df.to_csv('/Users/danfeldheim/Documents/OKC_project/vc_test2.csv')    
        
        
        return self.pbp_loc_player_distance_restricted_zone_df
   
       
        
    #----------------------------------------------------------------------------------------------    
    # Function for use with apply in the calculation of player distance variation coeficient (vc)
    
    def vc_calculator(self, a, b, c, d, e):
        
        number_list = [a, b, c, d, e]
       
        # Comment out the following and uncomment the section between dashed lines below to delete player distances > x ft.
        # Then remove rows with zeros in any of the vc column using R.
        mean = sum(number_list)/len(number_list)
        variance = sum([((x - mean)**2) for x in number_list])/(len(number_list)-1) 
        stdev = variance**0.5
            
        vc = stdev/mean  
        
        """
        #---------------------------------------------------------------------
        # Tried to improve the model by deleting player distances beyond x ft from rim, but model gets worse as distance is constrained
        # Delete distances > x ft
        number_list = [i for i in number_list if i < 30]
        
        # If statement in case number_list contains one or zero numbers
        if len(number_list) == 0 or len(number_list) == 1: 
            vc = 0
            
        else:
            mean = sum(number_list)/len(number_list)
            variance = sum([((x - mean)**2) for x in number_list])/(len(number_list)-1) 
            stdev = variance**0.5
            
            vc = stdev/mean  
        #---------------------------------------------------------------------
        """
        
        return vc
        
    #----------------------------------------------------------------------------------------------    
        
    # Function to calculate the number of offensive players that are in front of or to the side of their closest defensive player.
    # Hypothesis: Players who box out have a better chance of getting a rebound.
    
    def atRim_box_out(self):
        
        # Apply for players atRim (Rows from top to bottom below are x-y off, x-y def)
        self.pbp_loc_player_distance_restricted_zone_df['atRim_no_off_inFrontOf_def'] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.box_out_calculator
                (row.AtRim_loc_x_off_player_1, row.AtRim_loc_x_off_player_2, row.AtRim_loc_x_off_player_3, row.AtRim_loc_x_off_player_4, row.AtRim_loc_x_off_player_5, 
                row.AtRim_loc_y_off_player_1, row.AtRim_loc_y_off_player_2, row.AtRim_loc_y_off_player_3, row.AtRim_loc_y_off_player_4, row.AtRim_loc_y_off_player_5, 
                row.AtRim_loc_x_def_player_1, row.AtRim_loc_x_def_player_2, row.AtRim_loc_x_def_player_3, row.AtRim_loc_x_def_player_4, row.AtRim_loc_x_def_player_5, 
                row.AtRim_loc_y_def_player_1, row.AtRim_loc_y_def_player_2, row.AtRim_loc_y_def_player_3, row.AtRim_loc_y_def_player_4, row.AtRim_loc_y_def_player_5), axis = 1) 
                
    
        
        return self.pbp_loc_player_distance_restricted_zone_df
     
               
        
    def atShot_box_out(self):    
        
        # Apply for players atShot
        self.pbp_loc_player_distance_restricted_zone_df['atShot_no_off_inFrontOf_def'] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.box_out_calculator
                (row.AtShot_loc_x_off_player_1, row.AtShot_loc_x_off_player_2, row.AtShot_loc_x_off_player_3, row.AtShot_loc_x_off_player_4, row.AtShot_loc_x_off_player_1, 
                row.AtShot_loc_y_off_player_1, row.AtShot_loc_y_off_player_2, row.AtShot_loc_y_off_player_3, row.AtShot_loc_y_off_player_4, row.AtShot_loc_y_off_player_5, 
                row.AtShot_loc_x_def_player_1, row.AtShot_loc_x_def_player_2, row.AtShot_loc_x_def_player_3, row.AtShot_loc_x_def_player_4, row.AtShot_loc_x_def_player_5, 
                row.AtShot_loc_y_def_player_1, row.AtShot_loc_y_def_player_2, row.AtShot_loc_y_def_player_3, row.AtShot_loc_y_def_player_4, row.AtShot_loc_y_def_player_5), axis = 1)  
                
      
        
        
        
        return self.pbp_loc_player_distance_restricted_zone_df
       
                    
    #----------------------------------------------------------------------------------------------                     
    # Function for use with apply in the calculation of number of players that box out within n ft of the basket (start with n = 10)   
    
    def box_out_calculator(self, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
        
        # Create lists for off and def x and y coordinates
        self.off_x_coord_list = [a, b, c, d, e]
        self.off_y_coord_list = [f, g, h, i, j]
        self.def_x_coord_list = [k, l, m, n, o]
        self.def_y_coord_list = [p, q, r, s, t]
        
        # Find player distances from the basket
        # Create lists to hold distances from basket
        self.off_dis_from_hoop_list = []
        self.def_dis_from_hoop_list = []
        
        # Calculate distances and populate lists
        for x, y in zip(self.off_x_coord_list, self.off_y_coord_list):
            self.off_distance = ((x + 41.75)**2 + y**2)**0.5
            self.off_dis_from_hoop_list.append(self.off_distance)
            
        for x, y in zip(self.def_x_coord_list, self.def_y_coord_list):
            self.def_distance = ((x + 41.75)**2 + y**2)**0.5
            self.def_dis_from_hoop_list.append(self.def_distance)
   
        # Create dataframe with self.off_x_coord_shifted, self.off_y_coord, self.def_x_coord_shifted, self.def_y_coord, self.off_dis_from_hoop_list, and self.def_dis_from_hoop_list
        self.box_out_df = pd.DataFrame(list(zip(self.off_x_coord_list, self.off_y_coord_list, self.def_x_coord_list, self.def_y_coord_list, self.off_dis_from_hoop_list, 
                            self.def_dis_from_hoop_list)), columns = ['off_x_coord', 'off_y_coord', 'def_x_coord', 'def_y_coord', 'off_distance_from_hoop', 'def_distance_from_hoop'])
                            
        # Start counter for the number of offensive players that are:
        # 1. Within 6 ft of the basket
        # 2. Are equidistant or closer to the basket relative to any and all defensive players that are within 3 ft of the offensive player
        self.box_out_count = 0
        
        # Loop through each offensive player and calculate the number of offensive players that are closer to the basket than any defensive player that is within 3 ft of that offensive player
        for index, row in self.box_out_df.iterrows():
            
            # Get offensive player x-y coords and distance from hoop
            self.x_off_player = row['off_x_coord']
            self.y_off_player = row['off_y_coord']
            self.off_dis_from_hoop = row['off_distance_from_hoop']
            
            # Check to see if offensive player is within 10 ft of the basket
            # If yes continue, otherwise move to next player
            if self.off_dis_from_hoop <= 10:
                # Loop through def x-coord
                for index, row in self.box_out_df.iterrows():
                    # Check to see if offensive player is closer to basket than defensive player
                    # If yes, calculate distance between players, otherwise move on
                    self.x_def_player = row['def_x_coord']
                    self.y_def_player = row['def_y_coord']
                    if self.x_off_player <= self.x_def_player:
                        self.dis_between_players = ((self.x_off_player - self.x_def_player)**2 + (self.y_off_player - self.y_def_player)**2)**0.5
                        
                        # Check to see if defensive player is within 3 ft of offensive player
                        if self.dis_between_players <= 3:
                            self.box_out_count += 1
                            
                            
        return self.box_out_count
        
        
                    
                                           
    #----------------------------------------------------------------------------------------------    
    
    # Calculate average distance between offensive players and average distance between defensive players
    def player_spacing(self):
        
        # Create lists for final AtRim_off, AtRim_def, AtShot_off, and AtShot_def player spacings
        # These will be appended to self.pbp_loc_player_distance_restricted_zone_df
        self.AtRim_off_avgSpacing_list = []
        self.AtRim_def_avgSpacing_list = []
        self.AtShot_off_avgSpacing_list = []
        self.AtShot_def_avgSpacing_list = []
        
        # Calculate player spacing 
        for row in self.pbp_loc_player_distance_restricted_zone_df.itertuples():
            # Create temp list of lists for player x and y coord (gets populated with new values for every play)
            self.coordinates_list1 = [row.AtRim_loc_x_off_player_1, row.AtRim_loc_x_off_player_2, row.AtRim_loc_x_off_player_3, row.AtRim_loc_x_off_player_4, row.AtRim_loc_x_off_player_5]
            self.coordinates_list2 = [row.AtRim_loc_y_off_player_1, row.AtRim_loc_y_off_player_2, row.AtRim_loc_y_off_player_3, row.AtRim_loc_y_off_player_4, row.AtRim_loc_y_off_player_5]
            self.coordinates_list3 = [row.AtRim_loc_x_def_player_1, row.AtRim_loc_x_def_player_2, row.AtRim_loc_x_def_player_3, row.AtRim_loc_x_def_player_4, row.AtRim_loc_x_def_player_5]
            self.coordinates_list4 = [row.AtRim_loc_y_def_player_1, row.AtRim_loc_y_def_player_2, row.AtRim_loc_y_def_player_3, row.AtRim_loc_y_def_player_4, row.AtRim_loc_y_def_player_5]
            self.coordinates_list5 = [row.AtShot_loc_x_off_player_1, row.AtShot_loc_x_off_player_2, row.AtShot_loc_x_off_player_3, row.AtShot_loc_x_off_player_4, row.AtShot_loc_x_off_player_5]
            self.coordinates_list6 = [row.AtShot_loc_y_off_player_1, row.AtShot_loc_y_off_player_2, row.AtShot_loc_y_off_player_3, row.AtShot_loc_y_off_player_4, row.AtShot_loc_y_off_player_5]
            self.coordinates_list7 = [row.AtShot_loc_x_def_player_1, row.AtShot_loc_x_def_player_2, row.AtShot_loc_x_def_player_3, row.AtShot_loc_x_def_player_4, row.AtShot_loc_x_def_player_5]
            self.coordinates_list8 = [row.AtShot_loc_y_def_player_1, row.AtShot_loc_y_def_player_2, row.AtShot_loc_y_def_player_3, row.AtShot_loc_y_def_player_4, row.AtShot_loc_y_def_player_5]   
                
            # Create dataframe from the lists
            self.spacing_df = pd.DataFrame(list(zip(self.coordinates_list1, self.coordinates_list2, self.coordinates_list3, self.coordinates_list4, self.coordinates_list5, 
                            self.coordinates_list6, self.coordinates_list7, self.coordinates_list8)), columns = ['AtRim_loc_x_off', 'AtRim_loc_y_off', 'AtRim_loc_x_def', 'AtRim_loc_y_def', 
                            'AtShot_loc_x_off', 'AtShot_loc_y_off', 'AtShot_loc_x_def', 'AtShot_loc_y_def'])
                            
            # Create column counters
            # Note itertuples columns start with an index column so self.x_cntr must start at 1 here!
            self.x_cntr = 1
            self.y_cntr = 2
            
            # Create temp list of CLOSEST player distances from below; needs to refresh for each play so put inside this outer for loop
            self.closest_player_list = []
                            
            while self.x_cntr <= 7:
            
                # Calculate distance between all combinations of all players (e.g., start w/player and calculate distance to all other players)
                for row in self.spacing_df.itertuples():
                        
                    # Get player x, y-coords
                    # Get column names for use with itertuples
                    self.player_x = row[self.x_cntr]
                    self.player_y = row[self.y_cntr]
                   
                    # Create temp list of ALL player distances; needs to refresh for each player
                    self.player_dis_list = []
                    
                    # This inner loop calculates all combinations of player spacing    
                    for row in self.spacing_df.itertuples():
                            
                        # Calculate distance between players
                        self.player_spacing = ((self.player_x - row[self.x_cntr])**2 + (self.player_y - row[self.y_cntr])**2)**0.5

                        # Append to self.player_dis_list
                        self.player_dis_list.append(self.player_spacing)
                    
                    # Remove 0.0 from list (player's distance from himself). Need to do this outside of the for loop so that the list is not being modified while removing the 0.
                    self.player_dis_list.remove(0)
                    
                    # Find the closest player
                    self.closest_dis = min(self.player_dis_list)
                              
                    # Append self.closest_dis to self.closest_player_list
                    self.closest_player_list.append(self.closest_dis)
                    
                # Calculate average distance between players and append to master AtRim, AtShot lists at the top of the function
                self.average_distance = sum(self.closest_player_list)/len(self.closest_player_list)
                    
                # Append average player distance to final list
                # Need to do this based upon x and y counter numbers to populate the correct master list
                if self.x_cntr == 1:
                    self.AtRim_off_avgSpacing_list.append(self.average_distance)
                    
                elif self.x_cntr == 3:
                    self.AtRim_def_avgSpacing_list.append(self.average_distance)
                    
                elif self.x_cntr == 5:
                    self.AtShot_off_avgSpacing_list.append(self.average_distance)
                    
                else:
                    self.AtShot_def_avgSpacing_list.append(self.average_distance)
                    
                self.x_cntr += 2
                self.y_cntr += 2
      
                        
        # Append final lists to df
        self.pbp_loc_player_distance_restricted_zone_df['AtRim_off_spacing'] = self.AtRim_off_avgSpacing_list
        self.pbp_loc_player_distance_restricted_zone_df['AtRim_def_spacing'] = self.AtRim_def_avgSpacing_list
        self.pbp_loc_player_distance_restricted_zone_df['AtShot_off_spacing'] = self.AtShot_off_avgSpacing_list
        self.pbp_loc_player_distance_restricted_zone_df['AtShot_def_spacing'] = self.AtShot_def_avgSpacing_list
        
       
        
        return self.pbp_loc_player_distance_restricted_zone_df
       
        
        
       
    # Calculate the median distance between each offensive player and every other defensive player, not just the closer defensive player
    # This needs to be refactored to combine it with the player spacing function!
    def off_def_spacing(self):
        
        # Create lists for final AtRim_off, AtRim_def, AtShot_off, and AtShot_def player spacings
        # These will be appended to self.pbp_loc_player_distance_restricted_zone_df
        self.AtRim_off_def_medianSpacing_list = []
        self.AtShot_off_def_medianSpacing_list = []
        
        # Calculate player spacing 
        for row in self.pbp_loc_player_distance_restricted_zone_df.itertuples():
        
            # Create temp list of lists for player x and y coord (gets populated with new values for every play)
            self.coordinates_list1 = [row.AtRim_loc_x_off_player_1, row.AtRim_loc_x_off_player_2, row.AtRim_loc_x_off_player_3, row.AtRim_loc_x_off_player_4, row.AtRim_loc_x_off_player_5]
            self.coordinates_list2 = [row.AtRim_loc_y_off_player_1, row.AtRim_loc_y_off_player_2, row.AtRim_loc_y_off_player_3, row.AtRim_loc_y_off_player_4, row.AtRim_loc_y_off_player_5]
            self.coordinates_list3 = [row.AtRim_loc_x_def_player_1, row.AtRim_loc_x_def_player_2, row.AtRim_loc_x_def_player_3, row.AtRim_loc_x_def_player_4, row.AtRim_loc_x_def_player_5]
            self.coordinates_list4 = [row.AtRim_loc_y_def_player_1, row.AtRim_loc_y_def_player_2, row.AtRim_loc_y_def_player_3, row.AtRim_loc_y_def_player_4, row.AtRim_loc_y_def_player_5]
            self.coordinates_list5 = [row.AtShot_loc_x_off_player_1, row.AtShot_loc_x_off_player_2, row.AtShot_loc_x_off_player_3, row.AtShot_loc_x_off_player_4, row.AtShot_loc_x_off_player_5]
            self.coordinates_list6 = [row.AtShot_loc_y_off_player_1, row.AtShot_loc_y_off_player_2, row.AtShot_loc_y_off_player_3, row.AtShot_loc_y_off_player_4, row.AtShot_loc_y_off_player_5]
            self.coordinates_list7 = [row.AtShot_loc_x_def_player_1, row.AtShot_loc_x_def_player_2, row.AtShot_loc_x_def_player_3, row.AtShot_loc_x_def_player_4, row.AtShot_loc_x_def_player_5]
            self.coordinates_list8 = [row.AtShot_loc_y_def_player_1, row.AtShot_loc_y_def_player_2, row.AtShot_loc_y_def_player_3, row.AtShot_loc_y_def_player_4, row.AtShot_loc_y_def_player_5]   
                    
            # Create dataframe from the lists
            self.spacing_df = pd.DataFrame(list(zip(self.coordinates_list1, self.coordinates_list2, self.coordinates_list3, self.coordinates_list4, self.coordinates_list5, 
                                self.coordinates_list6, self.coordinates_list7, self.coordinates_list8)), columns = ['AtRim_loc_x_off', 'AtRim_loc_y_off', 'AtRim_loc_x_def', 'AtRim_loc_y_def', 
                                'AtShot_loc_x_off', 'AtShot_loc_y_off', 'AtShot_loc_x_def', 'AtShot_loc_y_def'])
                                
            # Create column counters
            # Note itertuples columns start with an index column so self.x_cntr must start at 1 here!
            self.x_cntr = 1
            self.y_cntr = 2
                
            # Create temp list of off-def player distances 
            self.off_def_spacing_list = []
                            
            while self.x_cntr <= 5:
                
                # Calculate distance between all combinations of all players (e.g., start w/player and calculate distance to all other players)
                for row in self.spacing_df.itertuples():
                            
                    # Get player x, y-coords
                    # Get column names for use with itertuples
                    self.player_x = row[self.x_cntr]
                    self.player_y = row[self.y_cntr]
                    
                    # Create temp list of ALL player distances; needs to refresh for each player
                    self.player_dis_list = []
                        
                    # This inner loop calculates all combinations of player spacing    
                    for row in self.spacing_df.itertuples():
                                
                        # Calculate distance between players 
                        # Use self.x_cntr + 2 to get the defensive players x and y coords
                        self.player_spacing = ((self.player_x - row[self.x_cntr + 2])**2 + (self.player_y - row[self.y_cntr + 2])**2)**0.5
    
                        # Append to self.player_dis_list
                        self.player_dis_list.append(self.player_spacing)
                        
                    # Get median value in self.player_dis_list
                    # Note this is the median spacing of one player
                    self.median_spacing_perPlayer = median(self.player_dis_list)
                    
                    # Append to self.off_def_spacing_list
                    self.off_def_spacing_list.append(self.median_spacing_perPlayer)
                    
                # Get median distance of all offensive players to defensive players
                self.median_spacing_allPlayers = median(self.off_def_spacing_list)
                    
                # Append median player distance to final list
                # Need to do this based upon x and y counter numbers to populate the correct master list
                if self.x_cntr == 1:
                    self.AtRim_off_def_medianSpacing_list.append(self.median_spacing_allPlayers)
                    
                else:
                    self.AtShot_off_def_medianSpacing_list.append(self.median_spacing_allPlayers)
                    
                            
                # Move counter to the 5th column (AtShot_loc_x_off)
                self.x_cntr += 4
                self.y_cntr += 4
        
        # Append final lists to df
        self.pbp_loc_player_distance_restricted_zone_df['AtRim_median_off_def_spacing'] = self.AtRim_off_def_medianSpacing_list
        self.pbp_loc_player_distance_restricted_zone_df['AtShot_median_off_def_spacing'] = self.AtShot_off_def_medianSpacing_list
        
        # Calculations verified to be correct 
        # self.pbp_loc_player_distance_restricted_zone_df.to_csv('/Users/danfeldheim/Documents/OKC_project/10_21_20_test_data1.csv') 
                    
        
        
        return self.pbp_loc_player_distance_restricted_zone_df
    
    # Function to include whether the top offensive and defensive rebounders are in the game and how many are in the game
    def top_rebounders(self):
        
        # Create list of top rebounders
        self.top_rebounders_list = [83, 275, 298, 81, 22, 108, 120, 170, 13, 26, 127, 823, 145, 717, 889, 341]
        
        # Set all NAs to '-' in the playerID columns
        self.pbp_loc_player_distance_restricted_zone_df = self.pbp_loc_player_distance_restricted_zone_df.fillna({'playerid_off_player_1': '-', 'playerid_off_player_2': '-', 
            'playerid_off_player_3': '-', 'playerid_off_player_4': '-', 'playerid_off_player_5': '-'})
        
        
        # Create a master list of lists (This will be converted to a df and merged with self.pbp_loc_player_distance_restricted_zone_df
        self.master_playa_dummy_list = []
        
        # Also get a count of the number of top rebounders in each play 
        # Create a list to hold the count
        self.no_top_rebounders_list = []

        for row in self.pbp_loc_player_distance_restricted_zone_df.itertuples():
            
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
        self.pbp_loc_player_distance_restricted_zone_df = self.pbp_loc_player_distance_restricted_zone_df.reset_index(drop = True)
        
        # Concatenate self.playa_df and self.pbp_loc_player_distance_restricted_zone_df
        self.pbp_loc_player_distance_restricted_zone_df = pd.concat([self.pbp_loc_player_distance_restricted_zone_df, self.playa_df], axis = 1)
        
        # Append self.no_top_rebounders_list to df
        self.pbp_loc_player_distance_restricted_zone_df['no_of_top_rebounders_in_play'] = self.no_top_rebounders_list
        
        
        
        
        return self.pbp_loc_player_distance_restricted_zone_df
          
     

    # Function to calculate the variation coefficient for player angles
    def angle_vc(self):
        
        # Get first column index 
        self.first_col = self.pbp_loc_player_distance_restricted_zone_df.columns.get_loc("AtRim_angle_off_player_1.0")
        self.second_col = self.first_col + 1
        self.third_col = self.first_col + 2
        self.fourth_col = self.first_col + 3
        self.fifth_col = self.first_col + 4
        
        self.x_cntr = 1
        
        while self.x_cntr <= 4:
            
            if self.x_cntr == 1:
                self.header = 'AtRim_off_angle_vc'
               
            # elif self.x_cntr == self.x_cntr + 5:
            elif self.x_cntr == 2:
                self.header = 'AtRim_def_angle_vc' 
                
            # elif self.x_cntr + 10 <= self.x_cntr <= self.cntr + 15:
            elif self.x_cntr == 3:
                self.header = 'AtShot_off_angle_vc' 
                
            else:
                self.header = 'AtShot_def_angle_vc'
            
            self.pbp_loc_player_distance_restricted_zone_df[self.header] = self.pbp_loc_player_distance_restricted_zone_df.apply(lambda row: self.vc_calculator
                (row[self.first_col], row[self.second_col], row[self.third_col], row[self.fourth_col], row[self.fifth_col]), axis = 1)
             
            
            self.first_col += 5
            self.second_col += 5
            self.third_col += 5
            self.fourth_col += 5
            self.fifth_col  += 5
            self.x_cntr += 1
            
            
        # Tested and calculations confirmed 
        # self.pbp_loc_player_distance_restricted_zone_df.to_csv('/Users/danfeldheim/Documents/OKC_project/vc_test2.csv')    
        
        
        return self.pbp_loc_player_distance_restricted_zone_df
   
    # Function to calculate the distance difference between every offensive player and their closest defensive player relative to the rim
    # For example, if offensive player is 3 ft from the basket and their closest defensive player is 4 ft from the basket, the difference is +1
    def distance_delta(self):
        
        # Find closest defensive player to each offensive player
        # Create final lists of delta player spacing (will be appended to self.pbp_loc_player_distance_restricted_zone_df)
        self.p1_delta_spacing_list = []
        self.p2_delta_spacing_list = []
        self.p3_delta_spacing_list = []
        self.p4_delta_spacing_list = []
        self.p5_delta_spacing_list = []
        
        for row in self.pbp_loc_player_distance_restricted_zone_df.itertuples():
            
            # Create temp list of lists for player x and y coord (gets populated with new values for every play)
            self.x_coordinates_list = [row.AtRim_loc_x_off_player_1, row.AtRim_loc_x_off_player_2, row.AtRim_loc_x_off_player_3, row.AtRim_loc_x_off_player_4, row.AtRim_loc_x_off_player_5, 
                                        row.AtRim_loc_x_def_player_1, row.AtRim_loc_x_def_player_2, row.AtRim_loc_x_def_player_3, row.AtRim_loc_x_def_player_4, row.AtRim_loc_x_def_player_5]
            self.y_coordinates_list = [row.AtRim_loc_y_off_player_1, row.AtRim_loc_y_off_player_2, row.AtRim_loc_y_off_player_3, row.AtRim_loc_y_off_player_4, row.AtRim_loc_y_off_player_5,
                                        row.AtRim_loc_y_def_player_1, row.AtRim_loc_y_def_player_2, row.AtRim_loc_y_def_player_3, row.AtRim_loc_y_def_player_4, row.AtRim_loc_y_def_player_5]
                        
            # Create dataframe from the lists
            # First 5 rows are offensive players, last 5 are defensive players
            self.spacing_df = pd.DataFrame(list(zip(self.x_coordinates_list, self.y_coordinates_list)), 
                                    columns = ['AtRim_loc_x', 'AtRim_loc_y'])
                                    
            # Calculate distance from rim for each player
            self.rim_dis = ((self.spacing_df['AtRim_loc_x'] + 41.75)**2 + self.spacing_df['AtRim_loc_y']**2)**0.5
            self.spacing_df['rim_dis'] = self.rim_dis
            
            # Add a column indicating player # from 1-10
            self.spacing_df['player_no'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            
            for row in self.spacing_df.itertuples():
                # Create temp dictionary to store closest defensive player number (resets for each offensive player)
                self.closest_player_dict = {}
                # Get offensive player's distance from rim
                self.current_player_rim_dis = row.rim_dis
                
                if row.player_no <= 5: 
                    self.off_player_no = row.player_no
                    self.off_player_x = row.AtRim_loc_x
                    self.off_player_y = row.AtRim_loc_y
                    
                    for row in self.spacing_df.itertuples():
                        if row.player_no > 5:
                             self.def_player_no = row.player_no
                             self.def_player_x = row.AtRim_loc_x
                             self.def_player_y = row.AtRim_loc_y
                        
                             # Calculate distance between the selected offensive player and defensive players
                             self.player_dis = (((self.off_player_x + 41.75) - (self.def_player_x + 41.75))**2 + (self.off_player_y - self.def_player_y)**2)**0.5
                             # Append player number and spacing to self.closest_player_dict
                             self.closest_player_dict.update({self.def_player_no: self.player_dis})
                             
                    # Find the key (player number) corresponding to the minumum value in self.closest_player_dict
                    self.closest_player_number = min(self.closest_player_dict, key = lambda x: self.closest_player_dict[x]) 
                        
                    # Calculate the delta rim distance for each offensive player and their closest defensive player
                    # Get rim distance of self.closest_player_number
                    self.rim_dis_closest_def_player = self.spacing_df.loc[self.spacing_df['player_no'] == self.closest_player_number, 'rim_dis'].item()   
                    self.rim_delta = self.current_player_rim_dis - self.rim_dis_closest_def_player
                    
                    # Append to correct player list
                    if self.off_player_no == 1:
                        self.p1_delta_spacing_list.append(self.rim_delta)
                        
                    elif self.off_player_no == 2:
                        self.p2_delta_spacing_list.append(self.rim_delta)
                        
                    elif self.off_player_no == 3:
                        self.p3_delta_spacing_list.append(self.rim_delta)
                        
                    elif self.off_player_no == 4:
                        self.p4_delta_spacing_list.append(self.rim_delta)
                        
                    else:
                        self.p5_delta_spacing_list.append(self.rim_delta)
                    
        # Append each list to self.pbp_loc_player_distance_restricted_zone_df
        self.pbp_loc_player_distance_restricted_zone_df['p1_delta'] = self.p1_delta_spacing_list
        self.pbp_loc_player_distance_restricted_zone_df['p2_delta'] = self.p2_delta_spacing_list
        self.pbp_loc_player_distance_restricted_zone_df['p3_delta'] = self.p3_delta_spacing_list
        self.pbp_loc_player_distance_restricted_zone_df['p4_delta'] = self.p4_delta_spacing_list
        self.pbp_loc_player_distance_restricted_zone_df['p5_delta'] = self.p5_delta_spacing_list
                
        return self.pbp_loc_player_distance_restricted_zone_df
   
    
    def logistic_reg(self):
        
        # Train 
        
        self.X = self.pbp_loc_player_distance_restricted_zone_df[['no_of_top_rebounders_in_play','actiondescription_dummy', 'AtRim_off_dis_vc', 'AtRim_def_dis_vc',
                        'p1_delta', 'p2_delta', 'p3_delta', 'p4_delta', 'p5_delta']]       
        self.y = self.pbp_loc_player_distance_restricted_zone_df['oreb']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.4, random_state = 1)
        
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.probs = self.model.predict_proba(self.X_test)[:, 1]
        # self.class_report = classification_report(self.y_test, self.probs)
        # print ('Classification Report: ')
        # print (self.class_report)
        
        # Print confusion matrix and accuracy
        # self.conf_matrix = confusion_matrix(self.y_test, self.probs)
        # print ("Confusion Matrix: ")
        # print (self.conf_matrix)
        # print ('Accuracy: ', metrics.accuracy_score(self.y_test, self.probs))
        
        false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(self.y_test, self.probs)
        print('roc_auc_score for Logistic Regression: ', roc_auc_score(self.y_test, self.probs))
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.probs)
        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle = '--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr)
        # show the plot
        plt.show()
        
        # Calculate logloss
        # self.reb_probs = self.model.predict_proba(self.X_test)
        self.log_loss = log_loss(self.y_test, self.probs)
        print('log loss: ', self.log_loss)
        
        
        """
        # Generate a ROC curve (a little nicer looking than the one above)
        logit_roc_auc = roc_auc_score(self.y_test, self.logmodel.predict(self.X_test)[:,1])
        fpr, tpr, thresholds = roc_curve(self.y_test, self.logmodel.predict_proba(self.X_test))
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('/Users/danfeldheim/Documents/OKC_project/Log_ROC')
        plt.show()
        """
        
        # Cross validate
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv = 10)
        print('Cross-Validation Accuracy Scores', scores)
        scores = pd.Series(scores)
        scores.min(), scores.mean(), scores.max()
            
        
# Call class
obj1 = rebound()

# Must call
imports = obj1.import_files()
merge = obj1.merge()

# Optional calls
prepare = obj1.prep()
position = obj1.player_pos_data()

# Must call
distance = obj1.distance_from_hoop()
restricted = obj1.restricted_players()

# Optional calls
angles = obj1.player_angles()
dis_variation = obj1.distance_vc_from_basket()
at_Rim_boxout = obj1.atRim_box_out()
at_Shot_boxout = obj1.atShot_box_out()
spacing = obj1.player_spacing()
off_def_spacing = obj1.off_def_spacing()
rebounders = obj1.top_rebounders()
angle_vc = obj1.angle_vc()
delta_dis = obj1.distance_delta()

# Logistic regression
reg = obj1.logistic_reg()




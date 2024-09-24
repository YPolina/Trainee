from pandas.api.types import is_numeric_dtype

class Validator:

    #Class initialization
    def __init__(self, data):

        self.data = data

    #Check of data dtype
    def check_integer(self, column):

        if (is_numeric_dtype(self.data[column])):

            print(f'All values in column {column} are numeric')
            return True
        else:

            print(f'Error: unexpected dtype in column {column}')
            return False
        
    #Check of positive values
    def positive_values(self, column, allow_zero = True):

        if allow_zero:
            if (self.data[column] >= 0).all():
                print(f'All values in column {column} are positive')
                return True
        else:
            if (self.data[column] > 0).all():
                print(f'All values in column {column} are positive')
                return True
        print(f'Error: unexpected values in column {column}. All values must be positive')
        return False
    
    #Check data range
    def check_in_range(self, column, min_value, max_value):
        
        if (self.data[column].between(min_value, max_value+1)).all():
            print(f"All values in column '{column}' are in range: [{min_value}, {max_value}].")
            return True
        else:
            print(f"Error: Values in column '{column}' are out of range [{min_value}, {max_value}].")
            return False
        
    #Check missing types  
    def check_missing(self):
        
        if self.data.isna().sum().sum() == 0:
            print("There is no missing values")
            return True
        else:
            print("Error: Missing values")
            return False
        
    #check duplicates
    def duplicates(self):
        
        if self.data.duplicated().sum().sum() == 0:
            print('There is no duplicates')
            return True
        else:
            print('Error: Duplicates')
            return False
    
    #Data check
    #1.id_check
    def check_id(self):

        id_columns = {'item_category_id':83, 'main_category_id':19, 'minor_category_id':66,
       'item_id': 22169, 'shop_id_encoded': 57, 'city_id': 29}
        for column, max_value in id_columns.items():
            if not (self.check_integer(f'{column}') and self.check_in_range(f'{column}', 0, max_value)):
                return False
        return True
    
    #2. Date_block_num
    def date_block_num(self):

        return self.check_integer('date_block_num') and self.check_in_range('date_block_num', 0, 34)
    

    #Date check
    def check_month(self):
        
        return self.check_integer('month') and (self.check_in_range('month', 0, 12))
    
    def check_year(self):
        return self.check_integer('year') and (self.check_in_range('year', 2013, 2015))
    
    #Revenue and price check
    def revenue(self):
        return self.check_integer('revenue_log') and (self.positive_values('revenue_log')) 
    
    def item_price(self):
        return self.check_integer('item_price_log') and (self.positive_values('item_price_log')) 
    
    #Validation
    def validate(self):
        
        if (self.check_id() and
                self.date_block_num() and
                self.revenue() and
                self.item_price() and
                self.check_month() and self.check_year() and
                self.check_missing() and self.duplicates()):
            return True
        else:
            return False
                                               


    

    



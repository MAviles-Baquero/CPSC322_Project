# Name: Tristan Call
# Assignment: PA4
# Date: 2/28/20
# Description: This mypytable is a table designed to hold a number of rows

import copy
import csv 
import math
import re
import mysklearn.myutils as myutils
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = len(self.data)
        cols = len(self.column_names)
        return rows, cols 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D structure storing data

        Notes:
            Raise ValueError on invalid col_identifier
        """

        # Get column index
        if isinstance(col_identifier, int):
            # If its an int, just copy it
            i = col_identifier
        else:
            # Else find from the header
            i = self.column_names.index(col_identifier)

        # Copy the data over
        col = []
        for row in self.data:
            if include_missing_values:
                # Include values with NA
                col.append(row[i])
            else:
                # Exclude values with "NA" or which are blank
                if self.is_valid_value(row[i]):
                    col.append(row[i])

        return col 

    def get_columns(self, col_identifiers, include_missing_values=True):
        """Extracts columns from the table data as a mypytable.

        Args:
            col_identifiers(list of str): string for column names
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            mypytable

        Notes:
            Raise ValueError on invalid col_identifier
        """

        # Get column indexes
        header = col_identifiers
        header_indexes = self.get_key_indexes(header)
        data = []

        # Copy the data over
        for row in self.data:
            new_row = []
            invalid = False

            for i in header_indexes:
                if not include_missing_values:
                    # Exclude rows with any values with any invalid values
                    if not self.is_valid_value(row[i]):
                        invalid = True
                        break
                new_row.append(row[i])
            
            if invalid == False:
                data.append(new_row)

        return MyPyTable(header, data)

    def is_valid_value(self, value):
        """Determines if a value is valid or not

        Args:
            value (obj)
        
        Returns:
            bool: true if valid, else false
        """
        if str(value) != "NA" and str(value) != "N/A" and str(value) != "":
            return True
        else:
            return False

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for j in range(0, len(row)):
                try:
                    value = float(row[j])
                except ValueError:
                    pass
                else:
                    row[j] = value

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        # For every row in rows_to_drop
        for row in rows_to_drop:
            # Attempts to delete that row
            # Only deletes 1
            try:
                self.data.remove(row)
            except ValueError:
                continue

        # compare entire row and if find identical delete. Coupled with find_duplicates. delete all matches

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")

        # Clear out all old data
        self.column_names.clear()
        self.data.clear()

        # Parse csv
        csvreader = csv.reader(infile)
        self.column_names = next(csvreader)
        for row in csvreader:
            self.data.append(row)
            

        infile.close()

        # Convert to numeric
        self.convert_to_numeric()

        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")

        # Write to csv
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(self.column_names)
        for row in self.data:
            csv_writer.writerow(row)

        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        duplicate_indexes = []
        header_indexes = []

        # Find header indexes
        for name in key_column_names:
            header_indexes.append(self.column_names.index(name))

        # For every row
        for i in range(0, len(self.data) - 1):
            row = self.data[i]
            try:
                # If the row is a duplicate do nothing
                duplicate_indexes.index(i)
            except ValueError:
                # For every row below that row
                for j in range(i + 1, len(self.data)):
                    lower_row = self.data[j]
                    # For every key if 1 doesn't match skip
                    for k in header_indexes:
                        if row[k] != lower_row[k]:
                            break
                    else:
                        # Otherwise the row is a duplicate
                        # Copy the index for future use and record the row
                        duplicates.append(lower_row)
                        duplicate_indexes.append(j)



        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        rows_to_remove = []
        for row in self.data:
            missing_value = False
            # For each value in each row see it it equals "NA"
            for value in row:
                try:
                    if value == "NA":
                        missing_value = True
                except ValueError:
                    pass
            # If there was a missing value record it
            if missing_value == True:
                rows_to_remove.append(row)

        # Then delete them all to avoid index failures
        for row in rows_to_remove:
            self.data.remove(row)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # Find the average of the column without missing values
        col_no_NA = self.get_column(col_name, False)
        ave = sum(col_no_NA)/len(col_no_NA)

        col = self.get_column(col_name)
        for i in range(len(col)):
            # For each value in the column if it is NA replace with the ave
            try:
                if str(col[i]) == "NA" or str(col[i]) == "N/A":
                    self.data[i][self.column_names.index(col_name)] = float(ave)
            except ValueError:
                pass

    def _compute_median(self, col):
        """Calculates the median of a 1D piece of data

        Args:
            col(1D list): continuous list
        
        Returns:
            median(float): the median
        """
        # Find the middle position
        col.sort()
        position = (len(col) + 1) / 2
        try:
            # If the position is an int, great!
            if position % 1 == 0:
                median = col[int(position) - 1]
            else:
                # Find the closest values and average them
                i = int(position // 1 - 1)
                median = (col[i] + col[i + 1]) / 2

            return median
        except ValueError:
            return None

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is 
                as follows: ["attribute", "min", "max", "mid", "avg", "median", "std"]
        """
        headers = ["attribute", "min", "max", "mid", "avg", "median", "std"]
        data = []

        for name in col_names:
            col = self.get_column(name, False)
            # Try to compute all the stats
            try:
                min_value = min(col)
                max_value = max(col)
                avg = sum(col) / len(col)
                median = self._compute_median(col)
                std = self.generate_std(col)
            except ValueError:
                continue
            else:
                # If successful make the row
                row = []
                row.append(name)
                row.append(min_value)
                row.append(max_value)
                row.append((max_value + min_value) / 2)
                row.append(avg)
                row.append(median)
                row.append(std)
                # Add to data
                data.append(row)

        # Make and return table
        return MyPyTable(headers, data=data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        table1_i = []
        table2_i = []
        header = []
        data = []

        # Construct header
        header = copy.deepcopy(self.column_names)
        for a in other_table.column_names:
            try:
                key_column_names.index(a)
            except ValueError:
                # If the attribute doesn't already exist, add to header
                header.append(a)

        # Attempt to find the appropriate indexes for each name in each table
        try:
            for name in key_column_names:
                table1_i.append(self.column_names.index(name))
                table2_i.append(other_table.column_names.index(name))
        except:
            # If this fails the tables aren't compatible
            print("Fail")
            pass
        else:
            
            for row1 in self.data:
                for row2 in other_table.data:
                    # Compare every row from table 1 with every row from table 2
                    equal = True
                    # See if all the key values are equal or not
                    for j in range(len(key_column_names)):
                        value1 = row1[table1_i[j]]
                        value2 = row2[table2_i[j]]
                        if value1 != value2:
                            equal = False
                            break
                    
                    if equal == True:
                        # Merge the rows by copying row1
                        row = copy.deepcopy(row1)
                        for k in range(len(row2)):
                            try:
                                table2_i.index(k)
                            except ValueError:
                                # Then append all non-key row2 indexes
                                row.append(row2[k])
                        
                        data.append(row)

        return MyPyTable(header, data=data) 

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        table1_i = []
        table2_i = []
        header = []
        data = []

        # Construct header with table1 columns
        header = copy.deepcopy(self.column_names)
        # Then add other table2 columns
        for a in other_table.column_names:
            try:
                key_column_names.index(a)
            except ValueError:
                # If the attribute doesn't already exist, add to header
                header.append(a)

        # Attempt to find the appropriate indexes for each name in each table 
        # relative to each key_column_name
        try:
            for name in key_column_names:
                table1_i.append(self.column_names.index(name))
                table2_i.append(other_table.column_names.index(name))
        except:
            # If this fails the tables aren't compatible
            print("Fail")
            pass
        else:
            
            for row1 in self.data:
                added = False
                for row2 in other_table.data:
                    # Compare every row from table 1 with every row from table 2
                    equal = True
                    # See if all the key values are equal or not
                    for j in range(len(key_column_names)):
                        value1 = row1[table1_i[j]]
                        value2 = row2[table2_i[j]]
                        if value1 != value2:
                            equal = False
                            break
                    
                    # If find a match, merge the rows
                    if equal == True:
                        added = True
                        # Merge the rows by copying row1
                        row = copy.deepcopy(row1)
                        for k in range(len(row2)):
                            try:
                                table2_i.index(k)
                            except ValueError:
                                # Then append all non-key row2 indexes
                                row.append(row2[k])
                        
                        data.append(row)

                # Add the row if doesn't already exist
                if added == False:
                    row = copy.deepcopy(row1)
                    # Add nas number of NAs to 
                    nas = len(header) - len(self.column_names)
                    for k in range(nas):
                        # Then append "NA" 
                        row.append("NA")
                    
                    data.append(row)
            
            # Add all the other row 2s
            for row2 in other_table.data:
                # Determine if the row exists already on key attributes
                match = False
                for row1 in data:
                    match = True
                    for j in range(len(key_column_names)): 
                        # Can get away with using table1_i on data row because
                        # the first attributes are copied from table1                       
                        value1 = row1[table1_i[j]]
                        value2 = row2[table2_i[j]]
                        if value1 != value2:
                            match = False
                            break
                    
                    # If a match is found, done
                    if match == True:
                        break

                # If no match is found, add one
                if match == False:
                    # Copy a random table one entry
                    row = copy.deepcopy(self.data[0])

                    # For every index in that row
                    for v in range(len(row)):
                        try:
                            i = table1_i.index(v)
                            # If the index is in the list of key table1 indexes
                            # Override with appropriate row2 value
                            row[v] = row2[table2_i[i]]
                        except ValueError:
                            # If the index is not in the list of key table1 indexes
                            # NA it
                            row[v] = "NA"
                    
                    # Then add the rest of row2
                    for i in range(len(row2)):
                        try:
                            table2_i.index(i)
                        except ValueError:
                            # If the value is not a key index from table2_i
                            # append it
                            row.append(row2[i])
                    
                    data.append(row)

        return MyPyTable(header, data=data) 


    def group_by(self, group_by_col_name, include_missing_values=True):
        """This function groups values of a column

        Args:
            group_by_col_name (string): The name of the attribute to group by
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            group_names (list of str):List of unique group names
            group_subtables (list of list of list of obj): list of sub tables for each group
        """
        # Find unique values
        col = self.get_column(group_by_col_name, include_missing_values)
        col_index = self.column_names.index(group_by_col_name)

        # Sets have no duplicates, so instant unique values
        group_names = list(set(col))
        try:
            # Try to sort the list if possible
            group_names = sorted(group_names)
        except TypeError:
            pass
        
        group_subtables = [[] for _ in group_names] # Produces [[], [], []]

        # For all rows put in appropriate column
        for row in self.data:
            value = row[col_index]
            # Find group if possible
            try:
                i = group_names.index(value)
            except ValueError:
                pass
            else:
                # Add if valid, non-NA value
                group_subtables[i].append(row) #shallow copy
            
        # Ensure all values are strings
        group_names = [str(name) for name in group_names]

        return group_names, group_subtables

    def get_key_indexes(self, key_cols):
        """This grabs the indexes for a list of key columns

        Args:
            key_cols (list of str or str)

        Returns:
            list of corresponding indexes
        """
        indexes = []
        if isinstance(key_cols, list):
            for key in key_cols:
                try:
                    # For every key add the corresponding instance
                    i = self.column_names.index(key)
                    indexes.append(i)
                except ValueError:
                    print("ERROR: Invalid key column")
                    raise
        else:
            try:
                # Do the same but if the input was not a list
                i = self.column_names.index(key_cols)
                indexes = [i]
            except ValueError:
                print("ERROR: Invalid key column")
                raise
        return indexes

    def extract_grouped_data(self, grouped_col_name, primary_keys, other_cols = []):
        """The purpose of this function is to handle the case of 'grouped data',
        where someone decided to stick multiples of data as 'action, comedy, etc'
        in one table instead of using multiple tables. It does this by making
        another table of the primary keys and the separated data that can be joined
        with the original for computation

        Args:
            grouped_col_name (string): name of grouped column
            primary_keys (list of str): keys to keep. NOT A STRING
            other_cols (list of str): non-key, non-grouped col columns to keep. NOT A STRING
        
        Returns:
            mypytable of form [primary_keys, grouped_data]
        """
        grouped_i = self.column_names.index(grouped_col_name)
        key_indexes = self.get_key_indexes(primary_keys)
        other_col_indexes = self.get_key_indexes(other_cols)

        # Construct header
        headers = primary_keys
        headers.append(grouped_col_name)
        headers.extend(other_cols)
        data = []

        # For all rows extract the genre
        for row in self.data:
            grouped_data = row[grouped_i]
            separate_data = re.split(',', grouped_data)
            for item in separate_data:
                # Then for every genre create a row with the primary keys, grouped data, and other cols
                new_row = []
                for i in key_indexes:
                    new_row.append(row[i])
                new_row.append(item)
                for j in other_col_indexes:
                    new_row.append(row[j])

                data.append(new_row)

        return MyPyTable(headers, data)

    def convert_percent_to_float(self, col_name):
        """Converts the selected column's values from percent form to float form
        Ignores empty values

        Args:
            col_name (string): name of the selected column
        """
        col_i = self.get_key_indexes(col_name)

        for row in self.data:
            percent = row[col_i[0]]
            # If there's a percent, remove it
            try:
                percent = percent.replace("%", "")
            except AttributeError:
                print("No percents to remove")
                break
            # Then try to convert to a float
            try:
                value = float(percent) / 100
            except ValueError:
                # If empty just ignore
                if percent == "":
                    continue
                # If not a percent trigger an error message
                print("Error: \'" + percent + "\' is not a percent")
                break
            else:
                row[col_i[0]] = value
        
    def generate_std(self, x):
        """This function computes the standard deviation

        Args:
            x (list of float): x values

        Returns:
            std
        """

        mean = sum(x) / len(x)
        smd = [((xi - mean) ** 2) for xi in x]
        variance = sum(smd) / len(smd)
        std = math.sqrt(variance)
        return std
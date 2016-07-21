
```python
#Python General commands

if __name__ == '__main__': #base code that will run when file is run by calling it by name
from __future__ import division

f = open(filename) #open filename
with open(filename) as f: #open file, prefer to use this over open(filename), so it automatically closes the file

class ClassName(object): #way to implement a class ClassName
  def __init__(self, selfvar1, selfvar2, function1, etc)
    self.selfvar1 = selfvar1

from collections import Counter #imports Counter "container" serves to count the most common items in a list/array  
Counter(array_lst_object).most_common(num_of_most_common_items) #returns a list with a tuple of the most common item and its count

if not os.path.exists("./<dirname>"): #to make a directory if doesn't exist using python
    os.makedirs("./<dirname>")

with open('URL here', 'r') as f:  #read URL location as file,
      shoes = f.read().replace('\n','') #reading it in as html for parsing using BS (see PyMongo below)

import requests  #ability to make GET request for web scraping
r = requests.get('http://www.ebay.com/.....complete URL here ')
soup2 = BeautifulSoup(r.content, 'html.parser')

from collections import defaultdict
from time import time #gives runtime, call with "print time()"

assign = _ #assigns var to the last output returned in the console
'-'.join(<sequence of strings>)


#PANDAS

%matplotlib inline # command in ipython to print graphs inline
import matplotlib.pyplot as plt #plotting functionality plt

pd.read_csv('filepath') # to read csv file into dataframe, assign to df name.
# try date_parser=[col index] parameter to allow specification of a specific column to parse as a datetime.
pd.read_table('text_filepath') #read text file as df
pd.scatter_matrix(<dfname>, alpha = 0.2, figsize=(6,6),diagonal='kde'); #scatter matrix of dataframe variables in pandas

df = df.drop('column name', axis=1, inplace=True)  # drops column in pandas dataframe, axis = 1 indicates column in pandas. default is axis = 0 which drops an observation (row) from the data
df.pop('column name') #pops off column name from df, modifying df in place
df = df[df['Column name']  <some value] # index based on column value, syntax here effectively deleting by re-saving as df
df[['column 1', 'column 2']] #index multiple columns
df2 = df.rename(columns={'int_col' : 'some_other_name'}, inplace= True) #renamecolumns
df.info() # type of each column in df
df.hist('column name') #plots histogram over column (or all columns if left blank)
<dateseries>.dt.month #to get month (or other element) from pandas datatime object
#when grouping by in pandas, can add "as_index = False" to move index to top.
series.unique() # to pull unique values of a given series
df.unstack() #will switch from stacked (row-wise data) to column-wise data by the inner-most grouping index.
df.values #return an np.ndarray of either the dataframe or the dataframe index
#if result of slicing is a 1D dataframe, this will be a (vertical) pd.Series, which can't be used to index. So .values will turn into a 1D array, whcih can be used to index.
df.values.astype(float) # returns values of a df as float
df.astype(float) #suffix astype allows conversion to float.
df.columns.values #returns an array of the column names of df

import statsmodels.api as sm  # to import statsmodels package for regression capability
sm.regression.linear_model.OLS(dependent_array, independent_dataframe).fit() #this shoudl be assigned to an <object> and then can be displayed using:
print <object>.summary()
plt.scatter( credit_OLS.predict(), credit_OLS.outlier_test()['student_resid']) #plot student residuals vs fitted y's to look for outliers
sm.qqplot(<OLS_model_object>.resid) #plot QQ plot of residuals of prevously fitted statsmodels object name.  
pd.get_dummies(array) #create dummy variables
pd.set_index('column_name', inplace=True) #set column as index
pd.groupby(by=None, axis=0) #group df by first argument
df.iloc[<integerindex>] #allows indexing by index location in pandas dataframe

df.to_csv('filename', index=False) #write to csv


# Numpy

np.linalg.norm()  # get the addition of two vectors
np.logical_and(array1==True, array2==True) #returns boolean array corresponding to array1/2 satisfying the conditions described
arrr[(arrr[:,0]>0.6) & (arrr[:,1]<0.4)] #setting slices of an array to conditions will return an array which meets those conditions OR
np.array(thresholds)[(np.array(tpr)>0.6) & (np.array(fpr)<0.4)] #can even index a target array (thresholds) according to boolean statements for TWO OTHER arrays (tpr, fpr) which are the same size/shape  as the target arrray (assuming of course that they reference the same data pionts)
np.argsort(array) #returns an array2 with the indices of the sorted input array, which you can use to iterate over sorted values.  
arr[np.random.randint(0, size)] #np.random.randint returns a random integer which in this case is used to index the array randomly, with replacement. size = len(arr)
arr.tolist() #return array values as list
arr = np.array(arr) # turn a list or other object into a np array
arr.shape #returns a tuple describing the shape of the array in (rows, columns)
np.newaxis '''usage'''#if arr is a 1D array (ie arr.shape = (3,)), arr[np.newaxis,:] will return a row vector shaped (1,3) whereas arr[:,np.newaxis] will return a column vector shaped (3,1)
arr[np.min(array, axis = 1)>0] #np.min returns booleans for the condition >0 along rows (axis = 1)
arr = np.append(arr, item) #append item to an array. Can even initialze array without specifying shape as:
arr = np.array([]) #initialize empty array
np.linspace(start,stop,num=50) #create a var from start to stop with num points.
balance['Married'] = balance['Married'].map({'Yes': 1, 'No': 0}) #mapping to change responses from Yes to 1.
np.logical_not(<condition>) #allows you to exclude things in <condition> without an if statement.



# Sklearn
from sklearn.cross_validation import train_test_split
train_feature, test_feature, train_target, test_target = \
train_test_split(features, target, test_size=0.3) # to make test/training set for cross validation


from sklearn.linear_model import LinearRegression
linear = LinearRegression() #instantiate the LinearRegression class
linear.fit(train_feature, train_target) #fit linear regression model using the object linear which is an instantiation of the LinearRegression class

from sklearn.linear_model import LogisticRegression
anyname = LogisticRegression()
anyname.fit(x[['col1','col2']],y) #fit logistic regression model, using col1,2,i number of features of x

from sklearn.cross_validation import KFold
kf = KFold(len(array), n_folds = 5) #create Kfold indices for cross validation on array use as follows:
for train, test in kf: model.fit(X[train], y[train]) #usage for Kfolds cross validation

from sklearn.preprocessing import StandardScalar
scale = StandardScalar()
scale.fit(training_data) #scale data

from sklearn.grid_search import GridSearchCV #example code of how to use grid search
param_grid = {'learning_rate':[0.1,0.05,0.02], 'max_depth':[4,6], 'min_samples_leaf':[3,5,9,17]}
est = GradientBoostingRegressor()
gs_cv = GridSearchCV(est, param_grid).fit(X,y) #grid-search takes (essentially) a raw or minimally parameterized estimator Class, and performs a search across the parameter_grid for the best parameter by the given scoring method. Will automatically assign the best parameters to the gs_cv estimator object once complete.
gs_cv.best_params_
from sklearn.grid_search import RandomizedSearchCV #perform a random grid search, allows searching through a random set of the hyperparameters to get a sense of where to pick them, thus will run faster than GridSearch.

from sklearn.cross_validation import cross_val_score
cross_val_score(estimator_object, X, y=none, scoring = <scoring type>, cv=None) #obtains a accuracy score by cross validation. Takes a parameterized estimator object, but this object does not need to be fitted to training data, as this will take training data (x and y if available) and perform cross validation (train test, train test K-fold times) to obtain a cross-validated score. You can set the type of score using scoring = <scoring type, below>, default scoring type is the simples appropriate score for the method, such as accuracy for classifiers or R2 for regressors; y lets you set labels for supervised learning, cv defaults to 3 fold CV, can set other integer
types of scores: http://scikit-learn.org/stable/modules/model_evaluation.html

# Matplotlib

plt.legend() #to have plot labels show as legend
plt.axvline(x=0.4) #plot vertical line
plt.axhline(y=0.6) #plot horizontal line
df.hist()

#plotting several sub-plots
def fig(digits):
    fig, _ = pl.subplots(nrows=10, ncols=10, figsize=(12,12))
    for i,ax in enumerate(fig.axes):
        ax.imshow(digits.images[i])
        ax.axis('off')
    plt.show()


# NLTK
nltk.download('stopwords') #downloads stopwords for nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()


# PyMongo
from pymongo import MongoClient
client = MongoClient()
# Access/Initiate Database
db = client['test_database']
# Access/Initiate Table
tab = db['test_table']

from bs4 import BeautifulSoup #import html parser BeautifulSoup
soup = BeautifulSoup(shoes, 'html.parser') #instantiate BeautifulSoup parser
img_array = soup.select('img.img') #select CSS elements "img.img" from html using BS


# MongoDB

Using Mongo - General Commands for Inspecting Mongo
  help                        // List top level mongo commands
  db.help()                   // List database level mongo commands
  db.<collection name>.help() // List collection level mongo commands.
  show dbs                    // Get list of databases on your system
  use <database name>         // Change the database that you're current using
  show collections            // Get list of collections within the database that you're currently using

Inserting
  Once you're using a database you refer to it with the name db. Collections within databases are accessible through dot notation.

db.users.insert({ name: 'Jon', age: '45', friends: [ 'Henry', 'Ashley']})
db.getCollectionNames()  // Another way to get the names of collections in current database

db.users.insert({ name: 'Ashley', age: '37', friends: [ 'Jon', 'Henry']})
db.users.insert({ name: 'Frank', age: '17', friends: [ 'Billy'], car : 'Civic'})

db.users.find()
    { "_id" : ObjectId("573a39"), "name" : "Jon", "age" : "45", "friends" : [ "Henry", "Ashley" ] }
    { "_id" : ObjectId("573a3a"), "name" : "Ashley", "age" : "37", "friends" : [ "Jon", "Henry" ] }
    { "_id" : ObjectId("573a3b"), "name" : "Frank", "age" : "17", "friends" : [ "Billy" ], "car" : "Civic" }

Querying
db.users.find({ name: 'Jon'})                 // find by single field
db.users.find({ car: { $exists : true } })    // find by presence of field
db.users.find({ friends: 'Henry' })          // find by value in array
db.users.find({}, { name: true })        // field selection (only return name)
db.users.findOne()   #helpful to find one record so u can see structure

Updating
db.users.update({name: "Jon"}, { $set: {friends: ["Phil"]}})            // replaces friends array
db.users.update({name: "Jon"}, { $push: {friends: "Susie"}})            // adds to friends array
db.users.update({name: "Stevie"}, { $push: {friends: "Nicks"}}, true)   // upsert
db.users.update({}, { $set: { activated : false } }, false, true)       // multiple updates

Imports and Cursors
    To import existing data into a mongo database one uses mongoimport at the command line. In this way mongo will accept a number of data types: JSON, CSV, and TSV.
mongoimport --db tweets --collection coffee --file coffee-tweets.json
    Now that we have some larger data we can see that returns from queries are not always so small.
    use tweets
db.coffee.find()
    When the return from a query will display up to the first 20 documents, after that you will need to type it to get more. The cursor that it returns is actually an object that has many methods implemented on it and supports the command it to iterate through more return items.
db.coffee.find().count()      // 122
db.coffee.find().limit(2)     // Only two documents
db.coffee.find().sort({ 'user.followers_count' : -1}).limit(3)  // Top three users by followers count

Iteration
  MongoDB also has a flexible shell/driver. This allows you take some action based on a query or update documents. You can use an iterator on the cursor to go document by document. In the Javascript shell we can do this with Javascript's forEach. forEach is similar to Python's iteration with the for loop; however, Javascript actually has a more functional approach to this type of iteration and requires that you pass a callback, a function, similar to map and reduce.
db.coffee.find().forEach(function(doc) {
    doc.entities.urls.forEach(function(url) {
        db.urls.update({ 'url': url }, { $push: { 'user': doc.user } }, true)
    });
});

Aggregation
    Aggregations in Mongo end up being way less pretty than in SQL/Pandas. Let's just bite the bullet and take a look:
db.coffee.aggregate( [ { $group :
    {
        _id: "$filter_level",
        count: { $sum: 1 }
    }
}])
  Here we are first declaring that we're going to do some sort of grouping operation. Then, as Mongo desires everything to have an _id field, we specify that the _id is going to be the filter level. And then we're going to perform a sum over each level counting 1 for each observation. This information is going to be stored in a field called count. What do we get back?
  We can also do more complicated stuff as well. Here's a query that returns the average number of friends users in this dataset by country. We need to access the country code field of the place field, but that is easy with an object oriented language like JS.
db.coffee.aggregate( [ { $group :
    {
        _id: "$place.country_code",
        averageFriendCount: { $max: "$user.friends_count" }
    }
}])



# Boto for AWS

import boto

# setting up aws access keys using json
with open('/Users/sf321092/aws.json') as f:
    data = json.load(f)
    access_key = data['access-key']
    secret_access_key = data['secret-access-key']

#open boto connection
conn = boto.connect_s3(access_key, secret_access_key)

#upload histogram to AWS S3
bucket = conn.get_bucket('testing-geoff')
#create a new key, which is akin to a new file in the bucket
file_object = bucket.new_key('geofffig.png')
#set key contents from filename in local path
file_object.set_contents_from_filename('/Users/sf321092/ds/gal/daily/high-performance-python/geofffig.png')

# command line code to log onto amazon instance from terminal, turns terminal into a terminal on the remote EC2 instance using ssh
ssh -X -i /Users/sf321092/.ssh/galkey.pem ubuntu@ec2-54-225-0-87.compute-1.amazonaws.com

# Command line code to write files to EC2 instance using scp
scp -i /Users/sf321092/.ssh/galkey.pem /Users/sf321092/ds/gal/daily/high-performance-python/geoff_script.py ubuntu@e<public instance id>:~


# SPARK!
import pyspark
sc = ps.SparkContext('local[4]') #initiating spark context locally using 4 cores
sc.parallelize(lst) #creates an RDD for local list lst
sc.textFile('sales.txt') #I believe this creates an RDD from a text file
  .map(function) and .filter() # are functions you call on the sc object that perform transformations to create new RDDs from existing RDDs.
  .count()  # is an action and brings the data from the RDDs back to the driver.
  .first() #an action that returns the first entry in the RDD
  .take(2) #an action that returns the first two entries in the RDD as a list
  .collect() #an action that pulls all entries in the RDD, requiring the entire RDD to fit into memory  

#word count example using Spark
sc.textFile('input.txt')\
    .flatMap(lambda line: line.split())\
    .map(lambda word: (word, 1))\
    .reduceByKey(lambda count1, count2: count1 + count2)\
    .collect()

    # Example Spark Commands
    Expression	Meaning
    filter(lambda x: x % 2 == 0)	Discard non-even elements
    map(lambda x: x * 2)	Multiply each RDD element by 2
    map(lambda x: x.split())	Split each string into words
    flatMap(lambda x: x.split())	Split each string into words and flatten sequence
    sample(withReplacement = True, 0.25)	Create sample of 25% of elements with replacement
    union(rdd)	Append rdd to existing RDD
    distinct()	Remove duplicates in RDD
    sortBy(lambda x: x, ascending = False)	Sort elements in descending order

    Common Actions
    Expression	Meaning
    collect()	Convert RDD to in-memory list
    take(3)	First 3 elements of RDD
    top(3)	Top 3 elements of RDD
    takeSample(withReplacement = True, 3)	Create sample of 3 elements with replacement
    sum()	Find element sum (assumes numeric elements)
    mean()	Find element mean (assumes numeric elements)
    stdev()	Find element deviation (assumes numeric elements)

#Spark local UI is at localhost:8080

#create local master node, run from inside master tmux session initiated by first command
tmux new -s master #first command
${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.master.Master \
-h 127.0.0.1 \
-p 7077 \
--webui-port 8080

#create local worker node, run from inside worker tmux session initiated by first command
tmux new -s worker1
${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.worker.Worker \
-c 1 \
-m 1G \
spark://127.0.0.1:7077

#start ipython notebook to interact with LOCAL spark cluster
IPYTHON_OPTS="notebook"  ${SPARK_HOME}/bin/pyspark \
--master spark://127.0.0.1:7077 \
--executor-memory 1G \
--driver-memory 1G



#Tmux, run commands in terminal
brew install tmux             # Install tmux with homebrew
tmux new -s [session_name]    # Start a new tmux session
ctrl + b, d                   # Detach from that tmux session
tmux ls                       # Get a list of your currently running tmux sessions
tmux attach -t [session_name] # Attach to an existing session (can use a instead of attach)
tmux kill-session -t myname #kill session
tmux ls | grep : | cut -d. -f1 | awk '{print substr($1, 0, length($1)-1)}' | xargs kill #kill all tmux sessions



'''



```

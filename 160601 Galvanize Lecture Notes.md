160701 K Means Clustering/Hierarchical Clustering
  1. Supervised vs Unsupervised
    - The loss function (described by the (y - y-hat), ie MSE etc) effectively "supervises" our learning of the relationship between y = f(x).
    - In unsupervised learning. Can still try to discern structure from the x data.
    - Unsupervised learning is particularly sensitive to selecting the right model, and deciding which data goes into the model.
  2. Clustering
    - Dividing data into distinct subgroups. Since we don't have labels, choosing the value of k is of particular importance, since you never know the truth and can't cross-validate to infer the best k
  3. K Means
    - We want within-cluster variation to be small
    - Want assignment of points to groups so within cluster variation is the smallest. So one way is to minimize the squared euclidean distance between all pairwise points
    - K means procedure:
        - start by randomly assigning each data point to a cluster.
        - computer the centroid of the cluster
        - re-assign the data points to the nearest centroid
        - recompute centroid
        - reassign data points...and repeat until no further change.
    - Techniques to choose K, or to "learn" k
        - Elbow Method: Computer within cluster variation (sum of squares) for several values of K. Thus there often will be a k after which the decrease in variation decreases. So you pick the elbow. But there is not always an obvious elbow on which to pick k.
        - GAP statistics
        - Silhouette Coefficient
    - Curse of Dimensionality
       - Since we're computing distances between points in k means, in high dimensional spaces, distances become far apart in high dimensions.
       - Amount of data needed to compensate for high dimensional data increases quickly
   4. Hierarchical Clustering
      - Algorithm: each poitn is its own cluster, merge closest clusters, and end when all points are in a single cluster.
      - don't have to choose k at the start, number of groups depends on where we cut the dendrogram
      - Several distance measures to consider using, resulting in different dendrograms. Most common are complete and average 

160630 Time Series
  1. Reading notes:
      1. One main goal is to account for the correlated nature of points, which explains their relative "smoothness" in time series data. So try to describe some mathematical model to describe the appearance of the data observed.
      2. Methods include using white noise,
          -moving averages: replaces a white noise series w-sub-t with an average of its current value and its immediate neighbors past and future.
          - Autoregressive- uses the two previous values
          - Random Walk with Drift -  random white noise movement with a drift (slope?) factor
  2. Time Series Lecture: analysis of experimental datat observed over time, usually equally spaced.
    - Conventional statistical methods are restricted given the correlation of adjacent time points; given assumption of independent and identically distributed data
    - Overal: find a pattern from past data at hand, and forcast into the future
    - Assumptions is that the pattern will continue
  3. Components of a time series
    - Trend - long run up and down of the series.
    - Cycle: upward/downward around the trend
    - Seasonal Variations: patterns following yearly patterns
    - Irregular Variation: remaining erratic movements that can not be accounted for
    - Goal is to estimate the trend, cycle, seasonal components of a time series so all that's left is irregular fluctuations
  4. Time Series Regression
    - Linear regression featurized time: discarding assumption of normalized residuals
    - Literally add a factor to the linear model for trend
    - Similarly model seasonal variable using seasonal or seasonal-dummy variables.
        - Model L seasons in SNsubT using L-1 indicator variables S1 through SL-1
    -  SSS10 Time Series Trend Series
    - Binning of time series data (past) is informed by the goal of forecasting. What do you want to predict?
    - Additive Decomposition: y = TR + SN + CL + IR , defined as trend, seasonal, cyclical and irregular factors.
        - Estimate SN t by grouping seasons and averaging and making it zero scale.
        - Estimate TR t by subtracting out seasonalized estimates , and fit a standard regression
        - Estimate CL t by removing the season and trend, and perform a moving average
            - The average of irregular fluctuations will appx be zero
        - Estimate IR t by looking at the residual after removing the main components
        - We assume that hte pattern continues into the future and that there is no pattern in the irregular component: therefore we predict IR t to be zero, thus the point estimate of IR at time t is zero.
        - Thus forecast of point estimate at time t is y = tr + sn + cl (all sub t).
            - Can give a prediction interval y +/- B sub (t, alpha), B is the error bound in a prediction interval
        - Key assumption of additive decomposition is additive variation over time. If this doesn't hold and you can't stabilize the variance, can try multiplicative decomposition
    - Smoothing methods
        - Simple exponential smoothing: Used when there is no significant linear trend (ie slope), but mean is changing over time
            - If the mean remains constant, then the standard linear model can be used, giving equal weight to all time points
            - If mean is changing slowly may better capture trend if we can give more weight to more recent observations than older observations, which is simple exponential smooothing.
            - Optimize value of alpha (which weights most recent data value)
            - SSS11 Exponential Smoothing
            - SSS12 Exponential Smoothing equation: So for more distant observations (x from current), the weight (alpha(1-alpha)^x) gets driven to zero, so more recent observations have a higher weight
            - alpha is a hyperparameter to optimize by minimizing sum of squared error SSE
            - SSS13 The error bound for more distant observations (farther from today) will get larger. Since when alpha is large, the error term (Tau -1) x alpha squared gets driven to zero
        - Hold's Trend Corrected Exponential Smoothing : allows for modeling both a linear trend (slope) that changes over time
            - Allows modeling of the mean and the growth rate (slope or linear trend) to change with time.
            - Allows us to take int o account growth rate, by adding a factor "b".
            - thus have both alpha and gamma hyperparameters
            - SSS14 Holt's Exponential Smoothing
        - Holt Winter's Exponential Smoothing : adds a component of seasonality to the linear trend/slope.
            - Similar to Holts, but it additionally takes into account an additional seasonal factor
            - Note that you ned to have data of enough seasons in order for this to work
    5. ARIMA:
        The Box-Jenkins Methodology applies autoregressive moving average models to find the best fit of time series based on past values.
        1. Approach:
          - SSS15 ARIMA model approach
          - Model Identification

          - The autocorrelation function is the correlation between the current time and the time x times ago. Uses pearson correlation coefficient
          - Partial auto- correlation, is the correlation between two separated points with the effect of the intervening points conditioned out. Almost like multivaraible linear regression with intervening time points as variables. SSS16 phone snapshot
        - See ipython notebook notes for examples of autocorrelation and partial autocorrelation plots SSS16 Autocorrelation/partial correlation plots
            - So note the first time point 0, correlation is 1 (since correlated with itself), but for autocorrelation, as you get farther out, correlation decreases.
            - For partial autocorrelation, after taking into account the first lag, the second lag actually has little additional correlation.
        - Stationary Time Series:
            - A time series for which the statistical behavior of a set of observations is identifical to that of the shifted set of observations for any collection of time points for any shift h (lag). This is a VERY STRONG ASSUMPTOIN.
            -  So the weak stationary assumption is that this holds for the first two moments : mean and covariance (with it self h-lags away). These are less reliant on time.
            - In reality, most time series are not stationary.
        - So how to modify the time-series to improve the approximation of stationarity
            - Detrending - constraint 1
            - differencing - constraint 1
            - transformation  - constraint 2
            - These are methods to try to turn nonstationary data into stationary data so we can model it using ARIMA, given the ARIMA asusmptions of stationary data
        - Use Autocorrelation (AC) or partial AC plots to compare the how well each method (detrending or differecing)
        - for deternding, plot the residuals on an autocorrelation plot to see whether you adequately removed autocorrelation
        - So keep examining autocorrelation plots to see whether you've adequately addressed the stationarity assumption using your methods above: deternding, differencing, transformation.
        - IN J&J example, needed to take log, difference and difference the seasonality (every 4th quarter), before we acheived statioarity.
         - NOte, detrending via linear regression may allow us to take into account multiple variables other than time in the modeling (since ARIMA can't do this).
         - Once we achieve stationarity, we take a look at the timepoint where decay is achieved at timepoint k. This will help determine how far back to consider correlation between observations.
    6. Autoregresive Models :
        - Autoregressive-po models is based on the idea that the present value of the series can be explained as a function of the p past values.  So use PACF plot to determine p, see slides.
   7. Moving Average Model:
        - assumes that white noise is combined linearly to form the observed data.
        - based on prior errors
        - Use ACF plot to determine q,
   8. Autoregressive Moving Average Model ARMA:
        - So ACF will identify the q parameter
        - PACF will identify the p parameter
   9. ARIMA :
        - Additionally takes into account for d, the number of differences needed to achieve stationary
        - This then provides the model by whcih you can predict future values (your model can) as a function of x-prior data points and coefficients  
    10. Seasonal differences can also be modeled in S-ARIMA.
    11. Fit the best candidate models and compare how models perform using AIC or BIC comparison
    12. After lab learnings:
      A. It sounds like the approach is as follows: view the AC and PAC plots to get a sense of the k-units lag from the PAC (p) and the AC (q) plots.
      B. Next step is to make data stationary: Can play with detrending, but ultimately, it seems differencing is the better technique to try to achieve stationarity of the data. However, be aware of overdifferencing
          - you detrend (often) by fitting an OLS to the data, and plotting ACF, and PACF plots to the residuals
          - second order differencing is the equivalent of a second order derivative of the data
      C. Once you've achieved stationarity of the data via differencing and or detrending, then you need to determine the p and q parameters by looking at the PACF and ACF plots and noting lag.
      - if you notice lag in either PACF or ACF plots, these will inform the remaining parameters p and q respectively in your ARIMA model, based on the number of k-lag days you observe as significant in the plots before they drop within the error bounds.
     - Note that your choice of the seasonality parameter L will seem to (i believe) inform the numbers you choose for d and maybe little p/q, since if L is 7, then a D of 1 I believe suggests a 7x1 differencing that is significant.
     - Less clear on how to pick P/Q relative to p/q and even D relative to d.
      D. Then plug all these parameters into ARMIA and then tune further based on ACF and PACF plots until adequate.





160629 Model Comparision summary
  1. RF :
    - Fast, works well out of the box
    - Sensible feature importances
    - Accurate
  2. Gradient Boosting
    - More accurate, but requires much more tuning of hyperparameters compared to RF
    - Partial dependency plots can try to get at the interpretability of the relative importance of the features
  3. Logistic Regression
    - More interpretable
    - Con: Intperetabilty becomes harder with categorical variables and making dummy variables, because everything is with respect to the baseline...even for OTHER variables
  4. Performance metric:
    - Choose based on priority to your application at hand: ie minimizing false negatives (ie churn), or other.
    - Recall
    - F1
    - Accuracy might have been bad in this situation because of the imbalanced classes (churn was 60%)
        - Accuracy may also be bad if there is a cost imbalance
  5. Leakage: an example would be somehow leaving target data in a derived variable within the data. Ie calculating churn using last ride taken, and leaving in last ride taken.
  6. Possible starter paramter ranges for grid search over Gradient Boosting Trees:
        learnign rate 0.1-0.01
        max depth 4-6
        min sample 3,17
        max features 1-0.1
        n_estimators 100
  7. BIG UNDERSTANDINGS:
    - cross validation is generally only ever used with training data. Use this to obtain an estimate of what your test error might be, without ever using your test data (until the very end) to train your model.
    - For grid search and cross_val_score, you only ever need to feed this training data. Never really feed it test data
    - Select best model after you've tuned your parameters by comparing cross_val_score on trainig data only.
    - Once you've selected your best model, THEN you can use that model (with set coefficients) on test data to obtain an estimate of test error. This generates the parameters of accuracy and or other relevant metrics that you can tell others about to describe your model.
    - Then, if desired, you can refit coefficients on the entire data set before using this to predict future data.



160629 Michelle Career lecture
  1. Resume ideas:
    Programming: Python, Pandas, Numpy, Scipy, sklearn, NLTK
    Visualization: Matplotlib, seaborn, plot.ly
    Databases: SQL, MongoDB
    Distributed Systems: Spark, MapReduce,


160628 Intro to Natural Language Processing
  1. Methods to featurize documents of text
      A. Bag of words: create a vector of word counts, regardless of punctuation, capitaliation
      - Order of words matter here
      - sklearn has a vectorizer fnction for bag of words
      - once documents have features vectorized, can compare documents
          - using euclidean distance, you look at vector-distance, which may not make intuitive sense in the sense of a vector of different length but pointing in the same direction
          - cosine similarity : looks at the direction of the vectors rather than length/distance. thus this is often a more useful classifier.
          - SSS8 Cosine similarity
      - Bag of words is naive, just word counts, with equal word having equal weighting. Sense that word counts should be adjusted based on frequency.
      B. "TF*IDF" : Term Frequency, Inverse Document Frequency:
          - Words in only in one document (rare), have higher weighting. Words found in every document have lower weighting.
          - Note, similar in equation to information/entropy from classification trees: ln(1/p).
          - Basically rank words based on amount of information gain
          - SSS9 Inverse Document Frequency
          - Often more useful than bag of words, sklearn has tfidf vectorizer
      C. Feature Engineering for text
          - Tokenize (to turn words into their category, perhaps changing posessives to original word, expanding contractions aren't to are not etc)
          - Remove stop words: removing words offering less information such as "the," there are standard lists of stop words.
          - Stemming : walked -> walk
          - Lemmatizing: people -> person. There are standard lemmatizers
          - N-grams: tokenizing several words together: "the cat", "cat in". Serves to expand the feature space. Takes into account word proximity
          -  Skip Grams: does N-grams, but skips x words. So categorizes proximity of words.
          -  Dependency/probabalistic parsing: takes into account word dependencies on each other, some grammatical/syntactic associations between words

160628 Naive Bayes Lectures
  1. Maximum A Posteriori : predicts the outcome most likely under the posterior distribution.
    - Given a prior distribution P(H), and the likelihood of observing some data X across a set of hypotheses, we predict the hypothesis that maximizes the probability of P(H given X).
    - Since P(X) does not depend on the hypothesis, MAP estimation works by maximizing the numerator P(X|H)xP(H).  
    - SSS7 Maximum A Posteriori
    - Conditional independence only holds between events A and B when C is true. It is a key assumption about the features in the Naive Bayes Model. A and B are not independent when C is not true.
    - Bayes is a Generative model:
      - Prior models are discriminative model, estimating the conditional distribution P(Y|X).
      - Generative models: estimates the joint distribution of P(Y|X) and P(X). So in addition to estimating the distribution of the target, is also estimates the population distribution of the features X.
      - This estimated joint distribution allows us to generate new synthetic data by sampling from the join distribution.
   - Naive assumption of Naive bayes: that each of the features are conditionally independent given Y. Though this is not always true, this does not generally change the MAP estimate.
  2. Applying Naive Bayes to Document classification
   - Multinomial event model. Assumes that a category of ie book type have a higher probability of predictors ie words, than other types.
     - The Prior Distribution is estimated from a sample or corpus of documents. The probability of any class is estimated to be those from the sample population.
     - The conditional distribution represent the frequency of each vocabulary word in each class of document. Estimated by counting number of times a word shows up in a given class, divided by the total length of documents in that class. (So frequency of words in a give class of document).
     - The naive bayes model for text classification is the product of the conditional probabilities of each word given a type of document and the probability of that document (the product over all words). And the hypothesized document class that maximizes this probability is what is predicted by the Naive Bayes model.
   - Risk of numerical underflow. Since each probability in Naive Bayes for a given document (ie 2000 words) is small, there is a risk that each probability is too small for a computer to easily represent.
      - Log transformation of the probabilities can get around this.
   - Laplace Smoothing. Sometimes new words are found in the test set did not appear in that class of the training set. Thus, the solution is to add 1 to each word's frequency, to get around this.
   - Unknown words: If a new words was not seen in any class throughout the trainign set, this can also cause a problem. Can add a generic unknown word to the vocabulary and give it a small probability.
   - In theory, can also use a number of distributions for Naive Bayes.
   - Naive Bayes Pros:
        - Good with wide data, with more features than observations.
        - Fast to train, and good at online learning
        - Simple to implement.
    - Cons:
        - Can be hampered by correlated features
        - Probabalistic estimates are unreliable when class estimates are correct given the naive assumption.
        - Some other models can perform better, with better accuracy.


160627 Web Scraping
  1. NoSQL databases
      - Not Only SQL: MongoDB is a flavor of NoSQL.
      - NoSQL paradigm is schemaless, which is an advantage in some cases
      - MongoDB is a document-oriented DBMS
      - Collections in Mongo are like dictionaries in python
      - JSON Javascript Object Notation, formalized formatting for their objects/dictionaries.
      - Can have data redundancies in documents (non-normalized data), as Mongo does not enforce normalized data
      - A change to database generally results in needing to change many documents
      - Since there is redundancy, simple queries are generally faster, but complex queries are often slower.




160624 Chat with Michelle

- To get valuation, ROT is to multiply seed amount by 5 (ie they got  20% of the company’s total estimated valuation)
- For A round, can multiply by 3-4, ie investors got 25%-30% of the company.
- But basically try to ask for the actual valuation or preferred share stock price to to calculate tax burden.


160624 Imbalanced classes
--> Ways around it:
  1. Cost sensitive learning
      - Thresholding: plot expected profit over each threshold, then can select the threshold with the highest profit
      - Some models have explicit cost functions that can be modified to incorporate classification cost
          - less frequently used
  2. Sampling methods
      - Undersampling: randomly discards majority class observations
          - Works, and can reduce runtime on large datasets, but discards potentially important observations.
      - Oversampling: randomly replicate minority class observations to balance the training sample
          - Doesn't discard information, but is likely to overfit.
          - SMOTE : Synthetic Minority Oversampling Technique: Generates new observations from minority class.
              - Basically takes a random KNN of the minority class
              - In the imaginary x1-x2 (can be extended to x-i) grid space for the two selected points, it will generate a random point somewhere between those two points, varying on all features x1-xi.
              - Create x amount of these points to "oversample" the minority class.
              - CV to tune the k value of KNN for smote
              - Sort of assumes minority class points are clustered near each other, and can run into problems when they are not.
              - If minority class points are also clustered around the majority class points, ddthe SMOTE procedure could definitely weaken the ability to classify.
              - So there are certainly limitations with this method
    3. Review of Metrics
      - F1 score: harmonic mean of precision and recall
      - ROC: If you know cost/benefit matrix, or a specific target precision/recall don't use AUC, since you can choose threshold to maximize PROFIT or those specific metrics.
      - AUC is more useful when exploring models to choose the best one


160623 Boosting
  1. Bias Variance Tradeoff:
      A. For regression trees:
      - Decrease bias by building deep trees
      - Decrease variance by using bootstrapped datasets such as in bagging, limiting features available at each node, pruning the tree, and majority vote ensembeling votes from various trees (averaging across trees decreases variance)
      B. Test methods of decreasing bias or variance by CV
      - In bagged decision trees, can use out of bag estimation to try to estimate external generalizability.
    2. Boosting Big Picture
      A. Attempts to make certain trees "become experts" in certain aspects of the data
      B. Boosting combines a sequence of weak learners to form a strong learner. Force it to be weak by limiting how deep it can be.
      C. Each tree learner fits error from the prior learner. Thus its an additive model of weak learners. "Forward stage-wise additive modeling"
    3. Ada Boost "Adaptive Boosting"
      - Each tree is expert on attacking error of its predecessor. So upweight misclassified points.
    4. Gradient boosted regression trees
      - Each successive tree fits to the residuals of previous tree.
      - Therefore each subsequent tree truly becomes an expert in the errors of the prior tree
      - And the final prediction is an additive sum of the "prediction" for each tree.
      - Gradient Boosting SSS1.
     - The weight beta-m in the sum of the predictions of trees, the weight is higher for later trees, since they can
     - Note this is prone to overfit, since we're fitting to the errors of prior trees. So use CV to determine when to stop.
     - Shallow trees are higher bias, lower variance models. (High bias since shallow, low variance because if we took another sample of data from the population, another 3 split tree would look similar)
     - So by limiting sequential trees to 3 splits, you effectively hone in on the highest error regions of the prior tree, making splits that decrease error (MSE) the most for any given tree.
     - ??What if we bag boosted models?
       - Random forest lends itself for running on parallel computers or cores on a computer. Sequential tree methods such as Gradient Boosting or AdaBoost can not take advantage of parallel processing.
   5. Parameter tuning
      - Could set a shrinking learning rate, "SHRINKAGE", meaning that we'll only apply 20% of what each subsequent tree suggests to do. Can prevent over fitting.
          - This will require higher n_estimators/trees, because it by definition learns much slower.
          - Ultimately will lead to lower test error with shrinkage. But requires more trees to get there.
          - Since this generally works: so one strategy is to tend to pick the max number of computationally possible trees to make (given time) and search over several hyperparameters discussed here via CV.
      - Parameter tune tree structure:
          - Max depth: controls degree of interactions between variables; often not larger than 4-6
          - Min samples per leaf: limits overfitting
      - Number of trees should also be tuned, because if you have too many, test error tends to increase, due to overfitting
      - Note that the SHRINKAGE parameter (learning rate in GB) is very important, and if not included (ie for stochastic gradient boosting below), can lead to unstable estimates because full votes from subsamples tend to go in a worse direction. so need to limit their vote
  6. Stochastic gradient boosting
      - So fit to a fraction of the total data at each step
      - Also fit a random subsample of features (like random forests)
          - In Sklearn's implementation of gradient boosting, max_features limits the features space explored for best split at each node.
      - These two techniques lead to both improved accuracy and decreased computational time
      - Often need to ensure a Shrinkage parameter for these to work.
  7. sklearn.grid_search
     - input dictionary of hyperparameters and a list of their options.
     - Will cross-validate search over permutations of parameters and return the best combinations of parameter.
     - SKlearn recommends: setting n_estimators as high as possible (3000,5000)
     - Tune all hyperparameters via grid search
     - Further optimize n-estimators and learning rate
     - Can review Rs's gbm package documentation
     - You must tune the model to get good performance
   8. Diagnostics
    A. Ways to infer feature importance from black box models
      - Partial dependency plots
        - Take your feature/variable x1 you want to study. Change all data points x1-i to x1-j to values x1i thorugh j sequentially. IE CHANGE ALL VALUES to x1-i once, then X1-i2 once etc.
          - record the predicted output y of your model
          - then can generate a plot for a given feature of how your model predicts y to change based on numeric changes of feature x1.
          - can repeat for all features and obtain a sense of the importance of each feature in your model. ie some inferential power.
          - can also do partial dependence plots for 2 variable interactions. to try to see if there is interaction by slope of plane?? not sure
      - Variable importance measures
          - Recall we discussed for bagged/RF trees, decrease in RSS for a given predictor averaged over all trees. Or change in Gini index for a given predictor, averaged.
      - Sensitivity analysis, where you sequentially change the value of a feature and see response change (similar to partial dependency plots)
  9. AdaBoost
      - Significance of a given tree to the final model is weighted by the amount of error it demonstrates.
      - So it effectively upweights trees that perform better, whcih will be later trees in many cases.
      - error is defined as the number of mis-classifications over total predictions
      - AdaBoost Formula SSS2
  10. XG Boost
      - For every given feature, keep only the percentiles of that feature, which greatly reduces the space over which the algorithm needs to decide where to split.
          - but you still keep all observations of the data.
          - so you get computational speed gains to grow a gree
      - This serves as a form of regularization, resulting in a slightly less complex model and less variance.





      ```
      ***MAKE A DECISION TREE OF THE HIERARCHY OF TREE BASED METHODS, FROM DECISION TREE TO BAGGING TO RANDOM FOREST TO BOOSTING.  TO BEST UNDERSTAND THESE METHODS
      ```



      ???max depth in boosting decreases variance?
      ???review tree HIERARCHY
      ???stochastic gradient boosting defined by max features and subsample only?





160622 Liga Data lecture
  1. Perform sensitivity analysis (change values of each data point for each variable and see the effect on output) to help determine the most important features. See sensivity_analysis on wiki. OFAT
  2. Concept of Dependent by Category DBC:
    - Of the top 10 or so variables, systematically create hierarchical interactions.
    - Maintain a matrix of the calculation of the importance of each of the interaction variables (ie via sensitivity analysis method above)
    - This can often help identify the most important features for a model (the interactions), since the  important interactions will often be from the most important features (independently) and aren't too deep (one or two, rarely 3+)
    - Can also update the matrix when new data comes in to perform a model "refresh" rather than a full retrain. Not sure, but this may also help to allow selection of other interactions based on the new data, by comparing the pre-post performance of the interactions performance on your matrix
    - Can help with a model in production to refresh faster without too much investment of time/effort
  3. The larger the gap between training and test performance suggests worse generalizability of the model. So try to create a gap-adjusted performance metric to account for this. Want models to be generalizable, so don't just consider test performance
  4. http://dmg.org/ for a list of pmml data mining software, analysis methods and pmml compatability, which can help with production, since pmml is a portable code standard?



160622 Support Vector Machines
  1. Support vector classifier
      - need to scale data
      - known as a "maximal margin classifier"
      - works based on defining a hyperplane to the data (only works with perfectly separable data) which has a normal vector w.
          - The dot product of the normal vector w with unknown point u: if the sign is positive or negative, it will classify the point to one side or the other of the hyperplane.
          - In order to make the hyperplane most generalizable, you want to maximize the distance between the nearest points and the hyperplane.
          - Do this by setting up a margin around the hyperplane and this effectively is a constraint.
      - How to you optimize an equation with a constraint? Several ways but LaGrangian? is one way.  
          - Essentially we want to solve for w, which defines the orientation of the hyperplane, whcih is the important part.
         - b defines the distance of the hyperplane from the origin (constant? for a given plane).
         - So the optimization equation solves for alphas, relative to y (labels) and x's (points).
         - alpha goes to zero for all NON SUPPORT vector points. Therefore in w = sum(alpha-i, y-i, x-i), alpha is only relevant for the points closes to the line/margin.
         - The solution for alphas allow you to solve for w (normal vector of the hyperplane) with the equation above.
         - Since x's and y's only matter for support vector points (since alphas for all other points go to zero), SVC can be a CONCISE REPRESENTATION  of a large data set.
         - The optimization equation, however, is hard/slow to solve. Which is a limitation.
         - There are inherent properties of SVT which limit overfitting
     - SVMs shine with moderately sized data, given its benefits, discussed below, but is slow/less optimal for very large data sets.
   2. Slack variable
      - Adding a slack variable which establishes a "budget" for how many points can be on the "wrong" side of the margin. Slack variable is "squiggle" in the lecture.
      - This allows for SVC to function with (real world) non separable data
      - Do this by changing the hyperparameter C in SVC in sklearn, which denotes the cost function. The smaller it is, the wider the margins are. The larger it is, it will converge on the smallest road
      - All points within the margin are now considered support vectors
      - Since w = sum(alpha * xi * yi), as the margins change, and different numbers of support vectors are used, the slope of the dividing hyperplane will change, since the slope is determined by the normal vector w.
      3. Kernals:
        - Kernals allow us to get the result of the dot product of vectors after being transformed to a higher dimensional space without needing to know what the transformation is.
        - the kernals transform vectors to an "infinite dimensional" space. thus it is possible to find a boundary that classifies any point, in theory.
        - In other words, it allows us to fit SVCs to very non-linear boundaries to classify data.
        - Note that it is possible to overfit data using this, ie drawing a boundary around every single data point = not generalizable.
        - Kernals most often used are: linear, polynomial of degree d and radial basis function (gaussian) 'RBF'
            - RBF is the commonly used kernal though prone to overfitting
        - As you increase gamma (in SVC in SKlearn), the variance  around each boundry decreases (more likely to overfit)
        - As C increases, model tries to not get points wrong or misclassified, so again, more likley to overfit.
        - A Grid Search: calculate validation accuracies (crossvalidation) for changing C and gamma hyperparameters, to identify the optimal values of the hyperparameters
      4. For multiple classes, create an SVM model for one versus all or pairwise (one versus one, pairwise).
          -Then predict based on majority vote based on all the models


160621 Random Forest
  1. Review of Decision Trees
      - In regression trees, the tree effectively performs axis-paralell splits, cordoning off the data into regions defined by lines paralell to the (x,y,z etc) axis.
      - This is inherently different from linear regression models which speak to their strengths in some settings
      - How to regularize decision trees: Pre and post pruning
        - Post pruning; after fitting trees deeply, decide threshold to prune back tree. - Ie one threshold is to prune back to minimum of data in a terminal node > the p avaialable features
        - Pree pruning: define a penalty term alpha where alpha * abs(#terminal nodes) is a penalty term added to the MSE, and minimize this equation. Cross validate to determine alpha. ** This is used most commonly **
        - * However, ensemble methods to build trees ie random forest will almost certainly do better than either pre/post pruning *
      - Recall that in classification trees, chose an error metric such as Gini or Entroy error functions, both of which perform very similarly. In regression trees, use RSS to calculate information gain.
      - Making predictions, at each terminal node, for regression: predict using average. For classification: predict the most commonly occurring class.
  2. Pros and Cons of decision trees
      - Pros: how we think, easy to explain
      - Cons: tend to overfit--do not strike balance of bias/variance tradeoff well, computationally intense
          -  Variance of a deep tree is high (since a second set of data produced from the same distribution will likely perform poorly); ie doesn't generalize well. Though bias is low for the training data (ie fits training data well)
          -  Less deep trees have higher bias
  3. Bagging: Bootstrap AGgregatING  
      - growing  a lot of trees and adding them together, which can serve to average away the variance (CLT)
      - Use bootstrapping such that trees grown will be slightly different (since data is slightly differet). Becaues otherwise, tree growing for a given set of data is deterministic, ie would make the same tree over and over.
      - Inherently paralellizable, can have multiple cores/computers building a set of trees
      - Recall that in each bootstrapped sample, 2/3 of the features are shared between the different trees grown from different boostrapped samples.
      - Out of bag error: side-effect of building model from boostrapped sample is that you have 1/3 of the data not used for each bootstrapped sample that can be used for test-error estimation (ie an inherent validation set).
          - Can use the 1/3 data and evaluate them using the trees not grown  from that boostrapped sample.
  4. Random Forests:
      - BUILDS ON BAGGING: so you still take bootstrapped samples of the data (typically the same number of samples as the original data set).
      - At each split, consider a random selection of m predictors at each split, in each tree. Typically n = sqrt(p). Ultimately tune p this with cross validation.
      - Leads to decorrelation of the trees, which leads to improved performance over bagging.  Serves to decrease variance of the total model.
      - RF Bias Variance SSS4
      - As you build more trees in RF, bias stabilizes and variance asymptotically declines. Thus it is better to build as many trees as possible. The real limitation is computational time. And there is ultimtely a point beyond which growing more trees is essentially futile. So test for this via CV.
    A. Feature Importance
      - How to determine?
        - Frequency of use as a split feature?
        - Record total amount of RSS decrease (or Gini index) for a given predictor averaged over all B trees. Larger value indicates importance.
        - To evaluate the importance of the jth variable, record OOB accuracy of the model. Then shuffle (permute) all values of the jth variable across all the samples. Re-compute the accuracy. The lower it gets, the more important the jth variable was.  Repeat for each var and rank the importance of the var.
        - Other ways (Sklearn methods): higher in the tree, or the expected fraction of data points that reach a node are both surrogates
    B. Bias Variance Tradeoff
        - Since variance is proportional to pairwise correlation p, and inversely proportional to the average of B-trees made in a RF, the RF procedure serves to decrease variance by averaging over B trees. p goes down and B goes up.
        - Variance RF = p * six^2 + ((1-p)/B) * sig^2
        - Information Gain SSS3
        - SSS6 Bias Variance Tradeoff
        - Averaging over many RF trees serves to decrease variance of the model.
    C. Tuning
        - Hyperparameter: parameter set ahead of time, what the structure of the model is. It is not learned by the model to achieve best fit. Often choose them via cross validation.
        - hyperparameter in RF is m, describing how many features are (radomly) available at each node when building each tree of the RF.
        - Select via CV
        - You can test several hyperparameters via CV, but RF has very many ~14, so adding each becomes more complicated. So likely need to decide some using heuristics like: for classification m= sqrt(p) or regression m= p/3.
        - Note that tree depth (another hyperparameter for RF), if too deep, RF can overfit.
        - Can use Feature Importance techniques above to try to select features of greatest importance to include when building trees. Becomes an issue if m is small, then this leads to decorrelated trees (good for low variance), but if unimportant variables >> important variables, model will not learn.  Again, reinforces why important to tune m via CV which should identify this.
          - Seems like the above also seems to be augmented if number of trees grown is low relative to number of features.
    D. Other Issues
        - Can weight predictor classes using a user-specified loss matrix. Ie  certain pieces of information are more important or prefer to err on false positive etc.
        - Missing data: many ways to deal with it. But consider setting to nonsensical value so you don't include "false" data in model.
            - Note, can use cross-validation to determine best way to deal with missing data from several candidate methods
        - Random forest lends itself for running on paralell computers or cores on a computer. Sequential tree methods such as Gradient Boosting or AdaBoost can not take advantage of paralel processing.


160620 NonParametric Learners
  1. K Nearest Neighbors
     - Sort the training points x by increasing distance from new point
     - predict the majority label of the k closest points; select k as a hyperparameter based on experimentation or prior knowledge.
     - if k is too small, tend to be overfit (model doens't generalize well), if k is too large, will be insensitive. the extreme case is that if you use the entire population n as k, you will always predict the majority class
     - general rule, can start with k = sqrt(n)
     - important to SCALE features, since we're calculating the distance between the features, we need to have them on the same scale, otherwise, that with the higher range will dominate the distance equation, making the other feature irrelevant.
     - Can implement weighted voting, where closer points are weighted more
   2. Problems
     - The curse of dimensionality: knn works well for lower dimensional spaces, but as d>5, becomes more problematic. Becuase the nearest "neighbor" become far away in many dimensions.
     - Predicting can be slow
     - Difficulty with imbalanced classes
        - Can consider weighting or sampling schemes to counteract this. Ie in a fraud detection case where you have 0.1% of the sample as fraud instances, and the rest are valid instances, you can downsample (randomly) the valid cases when training your model.
     - categorical features don't work well
   3. Decision Trees
      - Calculate the entropy of the dataset at the root node, "before the first split."
      - Perform a split of the data on every available feature (x), and calculate the entropy before and after. The Information Gain is entropy before - after. Then choose a split based on the highest information gain.
        - H(S) "entropy" = - Sum of probability(S) * log 2 probability(S)
        - Information gain = H(S) - sum
      - This method is top-down and greedy, making a decision based on the best information available at the time.
          - note that the "greedy" quality may mean that you do not make the decision that optimizes the overall performance. Will not revisit split decisions.
      - Gini index: another splitting measure similar to the entropy equation.
          - sum(probability(S)(1-probability(S))
      - The misclassification rate is also available in sklearn, but has a linear slope up to the maximum of 0.5, so in practice does not work as well as gini or entropy.
    4. Pros cons
      - The tree will fit every single observation, so it will have a tendency to overfit the data
      - Prune the true to address overfitting (in sklearn): min leaf size, max depth, purity (data points of the same class), gain threshold
      - if you use decision trees, you must prune the model; though practically speaking, we use ensemble tree methods (random forests) so pruning is not really necessary
      - Pros: low bias, simple to understand/interpret, nonparametric and non-linear, computationally easy to predict, deals with irrelevant features, works with mixed features
      - Cons: High variance, computationally expensive to train, greedy algorithm
      5. Cost Complexity pruning
        - Since decision trees will tend to overfit, a way to avoid this is to prune back the tree. One way is by cost-complexity pruning, which selects a tuning parameter alpha penalizing the subtree's complexity and fit to the training data.
        - Often, select alpha via cross validation, comparing the mean squared prediction error via cross validation for each alpha. Pick the alpha to minimize average error.
        - Then return the sub-tree for the chosen alpha.
      6. Both CLASSIFICATION tree and REGRESSION trees can be built.
        - The RSS is used as a measure of "information gain" with each split of a regression tree (continuous outcome).
        - the Gini Index or "Cross entropy" equations are used for classification trees, whcih basically classify the error rate before and after a given split (correlating with information gain), therefore allowing decision on what is the most "informative" split at a given node.
        - Tree methods can outperform linear regression when there is a highly non-linear and complex relationship between the features and the response.


160617 Gradient Descent
  1. Optimization
      - Find the min or max of a function: maximize likelihood or minimize square distance
      - Machine learning methods rely on optimizing a "cost function". Cost functions are the mathematical definition of your machine learning goal.
      - Intuition for gradient descent: If you take a guess and pick a point on the curve. You can calculate the derivative of the function from that point ( or the derivative as suggested by picking two close-points), you can then see the direction of the derivative. Depending on if you're maximizing/minimizing the function, you move in the respective direction based on the derivative, check it again at the next point until your derivative is near zero (which represents a maxima or minima).
      - The "gradient" is defined as the sum of the partial derivative of multiple variables. This is actually what you calculate in the above scenario for multiple variables.
        - the gradient is in the direction of the steepest ascent. The gradient points upward, so we descend down the gradient
        - in gradient descent, follow the line of the steepest descent on the cost surface to find the minimum
        - Per iteration through features of the model, you update the parameters simultaneously with respect to that gradient. In other words, you calculate the gradient once per iteration.
        - NB: Gradient descent does not work with LASSO, since not differentiable given absolute value cost function
          - Requires differentiable and convext cost function
          - Only finds global optimum on globally convex functions
          - Convergence asymptotically
              - Choices of convergence criteria: max number of iterations, change in cost function, magnitude of gradient
          - Performs poorly without feature scaling. Since if higher varaince for some variables, may take extra time. Can scale features back once optimization is performed
    2. Stochastic Gradient Descent
        - Use a subset of data
        - SGD computes cost function using a single different randomly chosen observation per iteration. On average, it achieves the same as GD, but may have more variability
        - SGD actually converge faster on average than batch GD, can oscillate around the optimum
        - SGD generally requires less memory and computation and is faster, so is generally preferred.  


160616 Logistic Regression
  1. Accuracy is the percent of observations correctly classified (TP+TN)/n
      - However, imbalanced classes will inflate accuracy
      - Doesn't reveal what kind of errors are being made.
      - Precision: True positives over those who were classified as positive.
      - F1 score (or F beta score) is a weighted harmonic mean of precision and recall. Rewards balanced precision and recall, rather than if it is unbalanced (if one is higher than the other).
      -  ROC Curve, plotting true positive rate and false positive rate.
          - Area under the curve.
      - SSS5 Classifier Metrics



160615 Cross Validation Regularized regression
  1. Bias variance tradeoff
       - A biased model centers around the incorrect solution. Bias can result from underfitting or from collecitng data poorly
       - High variance in the model can be caused by overfitting or by insufficient data
      - A model's expected squared prediction error will depend on the variability of y, and the variability of the training set used to train our model.
  2. Cross Validation : helps to choose the best model that performs well on unseen data
      - Split data into training and validation sets
      - We use cross valiation in order to get a sense of the error. However, build the final model on all of the data.
      - Leave one out cross validation: build n models on n-1 data points and calculate MSE on 1 remaining data point. Computationally expensive and high variance since the tests are so small. So large variance in the error.
      - Medium between LOOCV and 1 fold CV is k-fold CV, below.
      - Use training set to train several models of varying complexity
      - Then evaluate each model using the validation set: R2, MSE, accuracy etc.
      - Keep the model that performs the best over the validation set
    - As you increase complexity of the model, MSE of the training data will go to zero.
      - but this will also increase the MSE of the test data.
    - So you want to choose the optimal model complexity that optimizes test MSE, while maintaining the most parsimonious model.
    3. k-fold Cross Validation:
      - Randomize data set into k groups, train using k-1 folds.
      - Validate using the left out fold and record the validation metric for each validation such as RSS or accuracy
      - Average the validation results.
      - By increasing the number of folds, you decrease the variability of the mean squared prediction error of the model by averaging over the folds.
      * SO PRACTICALLY SPEAKING we often use cross-validation during model BUILDING: to optimize hyperparameters such as lambda in LASSO or Ridge, or to compare different versions of models. Once the model is built, we often FIT the model to the entire TRAINING data set, to fit the coefficients, and then compare performance on the TEST data set.
  4. To prevent model fitting due to too many features:
      - Metrics to compare models
          1. R squared is 1 - (MSS/TSS), 0 = awful, 1= awesome.
          - R2 essentially is the ratio of the sum of squared errors for the fitted model versus the null model with the null model predicting the average of the target (true) value, regardless of predictors.
              - However R2 will increase as you increase parameters, making it a poor comparator between different models.
              - R squared increases as you increase predictors, so can't really be used to compare between models.
          2. Adjusted R2 penalizes for number of predictors, making it a better comparator of accuracy between models. (n-1) / (n-p-1)
              - However Adjusted R2 tends to under-penalize complexity
          3. Information Theoretic Approaches: AIC, BIC
          4. Test, training set. Compare test set error between models.
          5. Cross validation. Average test error across all k-folds, and compare this between models of interest.
                - Cross validation makes no parametric or theoretic assumptions
                - Given enough data, can be highly accurate. is conceptually simple.
                - Cons: can be computationally intensive, if fold size is low, can be conservatively biased.



      - Subset selection:
          - Best subset: try every model combination and pick the best model (computationally intensive)
          - Stepwise: iteratively pick predictors to be in and out of model.
          Forward/backward stepwise selection.
          - Test between model candidate built by stepwise: AIC, BIC, Adjusted R2, Mallows C. Or can use CROSS VALIDATION
            - Better NOT TO USE RSS or R2.
            - Mallows Cp: Increases penalty for each parameter included. So needs to reduce the RSS faster than the penalty term
            - AIC, BIC, Adjusted R2 have similar rationale as Mallows Cp.
          - AIC/BIC are available across the gamut of parametric models, so these are the more common methods to use to choose a model.
          - These can be thought of alternatives to cross-validation to pick the best performing model. Cross validation can be performed on any model, whereas AIC/BIC may not be available for some models.
      - Regularization - TBD
      - Dimensionality Reduction - TBD
  5. Regularized Linear Regression
    - Shortcomings of OLS
    - In high dimensions, data is usually sparse : the curse of dimensionality
    - Linear regression can have high variance (ie tend to overfit) on high dimensional data
        ** This leads to the desire to want to restrict or normalize or regularize the model so that it has less variance **
  6. RIDGE REGRESSION - "L2 norm"
    - So regularization adds a penalty term that penalizes larger values of beta (penalizes large values more than small values of beta)
    - penalty is in proportion to the square of the beta coefficient, hence "L2" normalizing factor.
    - lambda is the penalty applied to the coefficients
      - higher the lambda the more the "bias" of the model. Labda = 0 is no different from standard linear regression.
      - notably, all variables need to be normalized, since lambda will apply penalty (large values of the coefficient) proportional to its scale. So variables must be normalized.
    - changes in lambda changes the amount that large coefficients are penalized
    - increasing lambda increases the model's bias and decreases variance
    - Use cross-validation to identify the ideal lambda that decreases the error of the model.
    7. LASSO - "L1 norm"
      - Instead of the penalizing the square of the coefficient, you penalize the absolute value of the coefficient, which has the benefit of driving some coefficients to zero (rather than near zero with Ridge)
      - Tends to set coefficients exactly equal to zero
      - Automatic feature selection, leads to sparse models
      - Ridge is computationally easier because it is differentiable
      - True sparse models benefit from lasso, dense models benefit from ridge
      -  Even in a situation where you might benefit from L1's sparsity in order to do feature selection, using L2 on the remaining variables is likely to give better results than L1 by itself.
    8. So use cross-validation to select lambda which optimizes error for either LASSO or Ridge  
        - Range of lambda (alpha in SKlearn) is 0 to infinity.
        - A lambda of zero in LASSO is identical to linear regression




160614 Linear regression
  1. Parametric model
      - Assumptions are strong, but so are conclusions. Models are simple, interpretable and flexible.
      - y hat indicates prediction of Y based on X=x
      - RSS grows with n, not by itself interpretable.
      - R2 proportion of variance explained by the model
          - R2 increases with features included into the model.
          - As you increase features, you risk overfitting data
          - Overfitting is learnign the data's signal and the noise, limiting generalizabiility to additional nonobserved data.
      - F test can be used to compare a model with another nested model
        - If model with missing predictor/s don't matter for prediction, F statistic will be small
        - You get a probability of F statistic which "MAYBE" can be interpreted as the p-value for the entire model??
      - If you have covariates that are not significant by p-value, remember that you may have correlation between coefficients and that interpretation of the meaning of the coefficients may be difficult.
  2.  Assumptions of a linear regression:
      1. Linear relationship between predictors and outcome
      2. Errors (residuals between predicted and true) are independent and identically distributed. And in time series, there is no correlation between consecutive errors.
      3. Constant variance of errors, aka homoscedasticity. Ie errors don't change over time, versus the predictions or versus any independent variable.
      4. Normality of the error distribution

  3. Troubleshooting.
      1. Multicolinearity: If two or more predictor variables are highly correlated with each other, the uncertainty in model coefficients becomes large. Affects the interpretability of coefficients
          - Can use a correlation matrix to look for pairwise correlations.
          - use VIF (variance inflation factors) for more complicated relationships. To try to figure out whcih variables are colinear
              - Run OLS for each predictor as a function of all other predictors. K times for k predictors.
              - Benefit is that it looks at all predictors together.
              - Rule of Thumb is >10 is problematic
          - remove (but make note of) any predictor that is easily determined by the remaining predictors.
      2. Outliers: When y is far from predicted y hat. Least Squares is particularly affected by outliers.
         - Residual plots can help identify outliers.
         - Influence plots can distinguish outliers from high leverage points.
     3. Heteroscedasticity:  the existence of more than one population of variability/variance. Ie residuals do not have constant variance.
        - Can test by plotting the residuals
        - If heteroscedasticity is present, can invalidate the assumption that modeling errors are uncorrelated and uniform.  
        - Solution may be to transform Y, ie log transformation of Y, or sqrt(y).
    4. Normality of Residuals: Linear regression operates under the assumption that residuals are normally distributed
        - Can check this assumption with a QQ plot.
        - If residuals are not normal, may transform the response.
    5. Non-linearity relationship between outcome and predictors.
        - Can consider adding a squared version of the predictor.
        - same as adding a higher order polynomial of a predictor to the feature space. Ie adding x^2 or x^4
        - Also spline, polynomial, step function, local regression, generalized additive models.  
        - Can capture nonlinear aspects of the relationship between parameters and y using the above methods: spline, polynomial etc
        - Can also transform the y feature space to try to address this.
    6. Mean Squared Error is proportional to Residual sum of squares. In fact it is RSS divided by sample size n.





160613
Linear Algebra
  1. matrix multiplication
      - For the given row-column target in the matrix, sum of  mulitply the elements of matrix1 row times matrix2 column.
  2.
      - Only square matrices are invertable
      - Eigenvectors/eigenvalues are values that do not change except for by a scalar through a transformation
      - A stochastic matrix represents the probabilty of goig from one state to another.  
      - Axis zero is column-wise, axis 1 is row-wise in numpy (opposite in pandas)
  3. Exploratory Data Analysis
      - Look at the distribution of individual features
      - Then look at bivariate plots
          - ie scatter with kernal density plots
          - pandas scatter matrix
  4. Linear regression
      - Cost function: usually is ordinary least squares. So you minmize the residual sum of squares. Minimizing the cost function in linear regression.
      - The error term or the "residual" is the difference between predictions and is assumed to be iid, and normally distributed with N( mean = 0 and variance).
      - Reliability of linear regression
        - R squared, coefficient of determination. Compares your model versus a model that is just the mean. A high R squared on its own does not imply a good model.
        - If features are correlated, this breaks the assumption that features are independent and errors are iid so this can affect validiy of the model.
        - F test compares model with null model. Shortcoming is that it doesnt' tell you which beta is unequal to zero.
        - Can perform hypothesis test on coefficient



160610
Multiarmed bandit
  1. Conjugate Prior  
      1. Conjugate Prior: Posterior is proportional to the likelihood x prior
      2. We use probability distributions (which integrate to 1) to model both the posterior and the prior in the Bayes Theorem equation.
          - The beta distribution is a known distribution that we will be using to model the PRIOR event for an event that has a likelihood of a binomial distribution.
          - We model the POSTERIOR distribution with the binomial distribution usually.
      3. Assumptions of the conjugate prior method are the same as for the binomial distribution: one of which is that the probability is Constant
      4. For the beta distribution, the alpha parameter is the number of conversions you had, the beta is the number of non-conversions (using the CTR example for a website A/B test)
  2. Traditional A/B testing
      1. Epsilon first testing: explore first for x tests, then use results
          - Pitfalls include: only after a set time, do you use data and pick the better performer, potentially losing money
  3. Multi-Armed Bandit
      1. Strategies to optimize exploration and exploitation, leraning as you go and making changes to behavior.
      2. Method of adaptive, reinforcement learning.
      3. Goal is to maximize return and minimize "regret" or using the sub-optimal option.
      4. Strategies
          1. Epsilon Greedy: Explore with some probability epsilon, often 10%
              - All other times we will exploit the option with the best performance
              -After we choose a given bandit, update the performance based on the result.
          2. Upper Confidence Bound UCB1
              Choose which ever bandit has the largest value, zero regret algorithm
          3. Softmax
              - choose a bandit randomly in proportion to its estimated valu. Tau is a parameter that controls the randomness of the choice
              - pA is the percentage of people who convert on site A
          4. Bayesian Bandit
             - Modeling each of the bandits with a beta distribution with the shape parameters of alpha = won, beta = lost
             - Then take a random sample from each bandit's distribution and choose the bandit with the highest value. Update with each new trial










160609
Statistical Power

  1.  power
      1. Def: Probability of rejecting the null hypothesis given that the alternative hypothesis is True
      2. How to improve power?
        - One way to improve the power of the test is to increase sample size, so the distribution under the presumed alternative hypothesis has less variation and thus the area to the right of the dotted line (alpha) is higher.
        - Less variable data
        - Change alpha (lower it)
        - A larger effect size (that you want to be able to detect with your experiment) would also increase the power of the test. Though experimentally a larger effect size is harder to achieve, ie it requires a higher burden of proof


160608
Hypothesis Testing

  1. Bootstrap: As long as your original sample is REPRESENTATIVE of the population, because the number of permutations of combinations of subsequent combinations (Drawn with replacement) of the samples in your bootstrap (n to the n, if n = size of original sample) is so large, the summary statistics of random variables will follow normal distribution.
      A. So you can use bootstrapping to empirically estimate confidence intervals for the summary statistic from your bootstrap sample
      B. So bootstrapping does not always narrow the confidence interval of the summary statistic, it can provide a confidence interval for statistics that don't have CLT to provide a 95CI.  
          1. ie " it is available regardless of how complicated the estimator is".  Since bootstrapping "estimates the sampling distribution of an estimator by sampling with replacement from the original sample"
          2. Used to estimate the standard errors and confidence intervals of an unknown population parameter
  2. Hypothesis Testing
      1. Type 2 error, loss of opportunity error, fail to reject H0 when it is false
  3. Chi square
      1. Chi Square Goodness of Fit: Used to compare the sample data of a categorical variable to the theoretical distribution. Observed minus expected
      2. Chi Square Test of Independence: Compare two categorical variables under the assumption that they are independent  



160607
Sampling and Estimation

  1. Parametric vs Non Parametric
      A. Parametric based on assmptions about the distribution of the underlying population, that that there are parameters that define that distribution
          - If data deviates from these assumptions than a parametric procedure can lead to incorrect conclusions
      B. Nonparametric: does not make distributional assumptions about he shape or parameters of the underlying distribution
          - If base assumptions hold, parametric approaches can lead to more precise estimations
          - Usually based on ranking, does not take into account distribution of data
          - Less power than corresponding parametric procedure
          - Interpretation can be more difficult than parametric measure
    2. Maximum Likelihood Estimators
        A. The join distribution is the product of the individual probabilty distributions, this is ASSUMING that events are indepdent and identically distributed (IID)
            - Maximizing the likelihood function is the same as maximizing the log lilelihood function which simplifies complications, since the product of function is the same as the sum of the logs of the function. So calculation is simplified.
            - So derivative with respect to the parameter (partial derivative) is how you solve for this
      3. Maximum a Posteriori -MAP
          A. Baysean suggests that the parameters from come a distribution themselves. Assume a prior distribution of the parameters g over theta, given the value x.
      4. Kernal Density estimations
          A. Drawn from a distribution with an unknown density f, but which you are interested in estimating the shape of its function. Essentially a data smoothing problem.
          B. Uses a kernel K(.): which is a nonnegative function that integrates to one and has a mean of zero
          c. h is a smoothing parameter, called a bandwidth, which determines the width of the bins over which the kernel is deployed.
          D. Related to histograms, but can be made smooth by using a suitable kernel.  Sum of kernals, can be thought of as summing kernels




160606
Probability
- Probability Mass Function PMF:  For a DISCRETE random variables X that takes discrete values, give the probability of an individual event.
- Probability Density Function PDF ("CDF" in scipy stats modules): For CONTINUOUS random variables, gives probability of having value less/greater than a given value.

- disjoin is mutually exclusive
- upside down A is “all"
- Combinatorics
    - Factorial is the all the possible orderings in a set of items n
    - Permutations, selecting subgroups of a set when order matters
        - nPk = n! / (n - k)!
    - Combination: number of was of selecting subgroups when order doesn’t matter
        - nCk  = n! / (n-k)!k!
        - Note that you’re dividing out the number of orders (factorial k) in the number of the set chosen
    - Sample space contains the exhausted list of mutually exclusive simple events
        - An event A contains a subset of the simple events in S
    - Conditional Probability: probability given something already occurred. Reduces the sample space to the probability of what happened (what is given)
- Random Variable
    - Exepected Value: is the most likely outcome of a random variable
    - For discrete random variable, sum over the sample space; for continuous, need to integrate over sample space
    - Variance: measure of the spread of values around the mean; squaring this is how to make it positive (one method only of several), but the units are squared, so make it harder to interpret
        - Standard deviation is the square root of Variance, with units back to original variable
    - Covariance: how two variables vary in relation to each other. Ranges from negative to positive infinity. So the covariance re-scaled ranges from neg 1 to 1.

- DISCRETE DISTRIBUTIONS
- Bernoulli distribution: two outcomes, success failure
    - Constant Probability of success, events are independent
- Binomail models the sum of the Bernoulli random variable; so the number of successes in n trials
    - probabiltiy is the product of the independent Bernouli random variables, since each is independent
- Geometric distribution also builds on Bernouli, models the number of trials to first success
- Poisson, models nubmer of events/successes in a period of time (or space)
    - lambda is an average rate of success in that time or space
    - Good for counting processes

- CONTINUOUS DISTRIBUTIONS
- Uniform: equally likely events within a set interval
- Exponential: models time between Poisson events

160603
SQL Python

- psycopg library to interface to a Postgres database from within Python
    - There are Python libraries to connect to almost any database you want, mysql-connector-python, sqlite, pymongo
- Cursor is a control structure to fetch data and handle transactions with the SQL database
    - The results from a cursor object can only be accessed once, since it is returned as a generator.  So need to dump it into a datastructure in Python you can use later
    - Generally will use cur.fetchall() into a python object to use the results of your query
        - Can also use: cur.fetchone() #next result, cur.fetchmany(n) #returns next n results (in case you need to batch the storage of your results)
        - Or: for res in cur: #iterates over results in the cursor
- Enter SQL postgress query as a multiline string in python: query  = ‘’’      ‘''

- If you execute a query in psycopg2 and there is a mistake in your query, you can’t interact with the cursor again (ie send a fixed query) until you rollback the connection: conn.rollback()
- To open a connection in psycopg2:
    - import psycopg2
    - conn = psycopg2.connect(dbname=‘{name}’, user=‘[username]’, host=‘[hostname])
- Write queries in python using: c.execute(‘’’   SQL QUERY   ‘’')
- Changes to the database made using your query are NOT stored until you commit them using
    - conn.commit()
    - curr.close()
    - conn.close() #good practice to close the connection at the end of your python program
- conn.rollback() # if you make a mistake on the query, use rollback to restart the transaction.
- To create a new DATABASE in python through psycopg2, you need to turn on auto-commit in order to execute the command. Be sure to close the connection and restart a new one with auto-commit off, so you don’t accidentally change the database with a subsequent query that you can’t roll back.
    - conn.set_session(autocommit = True )
    - cur = conn.cursor()
    - cur.execute(‘CREATE DATABASE <database name>’)
    - curr.close()
    - conn.close()
- COPY <into table name> FROM ‘filepath of data file ie csv’ DELIMITER ‘,’ CSV; #import data into table from external file.

160602
SQL

- RDBMS data is one way to store persistent data
    - Data that is infrequently accessed and unlikely to be changed
    - Composed of tables, columns rows
        - Column contains a certain data type
        - Row is an observation that holds values for each of the columns
        - Tables are specified by a schema that defines the structure of the data
- Advanced SQL
    - Self-join-effectively joining a table with itself, in order to filter more powerfully. Used commonly.
    - A With function in SQL, you can alias a query, then use than in a larger query
    - Window functions is signified OVER ( PARTITION BY    ) , and allows a calculation over a set of table row within a query

160601
Object Oriented Programming

- A class is a blueprint of an object that can be created from that class, works generically across different cases. Ie a dictionary is a class in python
- An object is an instance of a class, can create multiple instances of the same class
- An attribute is a property of the class, usually a variable
- OO revolves around three concepts
    - Encapsulation - interact with internals of an object only through method calls, rather than directly
        - attributes of the object are essentially the data stored in the object (instance of the class), but only alter/update them through methods
    - Inheritance - Derive a child class from a base class. Base class defines general behavior. Child class specializes behavior.
    - Polymorphism - Objects of the same base class will often support the same interface, having the same/similar attributes or methods
- “self” refers to an instance’s own unique data.  When you assign a class to a variable, that variable subsequently then gets passed to every method of that class-instance as the first argument. So every method in that class must have a “self” argument in the list
- __init__ is the first method called in a class that is called whenever the class is created. Use self to refer to the instance’s member data and functions
- *args will pass a list sequentially to a function
- ** kwargs are keyword-named arguments that are stored in a dictionary and can be passed to a function, either as a dictionary or as individual named key-word arguments. preceeded by ** (packing and unpacking)
- A static method is a function in a class that is not passed the reference the instance of the class. So it’ll just do that it says, regardless of the rest of the class functions
- Magic methods: predefined methods within python that can be defined for your class, such as length or str(stringmethod)
    - repr is the python representation of the object
    - str is the string representation of the object
    - init is the constructor magic method required to initialize a class
- An Abstract Base Class defines a standard interface for derived object. So it allows you to define a common set of methods for a class-type, essentially enforcing polymorphism
    - Ie for any regression, they offer a certain types of methods. So define ABC first, then individual classes from this later
- Decorators: functions which wrap other functions
- Python Debugger is a good way to debug clode
    - insert “import pdb” in line in the code,
    - “import pdb; pdb.set_trace
    - type n for new line, c for new trace point, q to quit, s steps through news line stepping through functions
    - can access anything at that python runtime, paused at the given line

160531

- Programming
    - Generators in Python
        - Take less memory
        - Use enumerate when you need the index of the list as well as the value
    - If possible, try to see if you can use hashing to organize/loop over data, as this is the fastest way to perform such functions
    - Mutable objects can change their value but keep their id()
        - lists, sets, dictionaries
        - Only immutable objects can be used as key in dictionary (“unhashable”)
        - Immutables: int, string, tuple, floats

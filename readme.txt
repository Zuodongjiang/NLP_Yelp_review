1. Description:

We use Ruby on Rails as our web framework. D3 has been included to produce the graph visualizations. Postgresql is used as our database to store the data. In data processing, we used python-NLTK to pre-process text data, Gensim to calculated word embedding and Tensorflow/Keras to run classification on sentiment polarity score with LSTM model. Finally, we use KNN model to aggregate the score with the chek-in (percentile) rankings for certain geographic loacation and generate positive/negative keywords using TF-IDF.

2. Installation:

2.1 Ruby and Rails
First, check if you have Ruby already installed by typing `ruby -v` in your terminal. If you get an error, visit rubyinstaller.org, click download link, and install Ruby.

Then, check if you have Rails already installed by typing `rails -v` in your terminal. If you encounter an error, visit http://installrails.com/, click Start Now and follow the instructions to install Rails.

Now, you should have ruby and rails installed. Navigate to the project folder root directory and type `bundle install` for installing the framework dependencies.  Type `rake db:create && rake db:migrate && rake db:seed` to create a database, and set up the schema.

2.2 Python Environment

We recommend you to used the Deep Learning Amazon Machine Image (https://aws.amazon.com/marketplace/pp/B077GCH38C) with pre-installed/configured machine learning packages required for this project, including NLTK, Gensim, Tensorflow and Keras. 

Also, check if have psycopg2 installed. If not use conda install or pip install psycopg2 to install the package for postgresSql connection. 

3. Execution:

Type `rails s` to spin up the server. Open your browser and type `localhost:3000/something/new`, you should be able to see the welcome page and perform corresponding actions.

To save your time, we have set up a demo environment for testing on http://cse6242.herokuapp.com/something/new.

It may take sometime to spin up the server as we are using a free tier, but the environment acts the same as setting up the code locally.

Please make sure you are using http instead of https, as our free tier will block the Javascript assets for https connection so the D3 graphs will not be shown.

To reproduction the word embedding calculation, run Word2Vec_large.ipynb file in the environment mentioned above; Then trigger train model.ipynb to run the LSTM model and load the data to postgresSQL server. Finally, run prepare_output.py to run KNN model to produce the result.
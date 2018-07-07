# 100-Days-of-ML-Code
A little side-project to try to classify monologue according to their "moods."

The initial file, moodScraper.py will scrape all posts from relevant subreddits that are classified as "mood" subreddits (i.e. depression, angry) and save all posts to local directories, where posts are saved under their id, and comments are 'tagged' for parsing later on.

The next step will be to apply word2vec to effectively vectorize all the words gotten from the posts, which will later be used in training a machine learning algorithm to classify text based on its mood.

Once the words are vectorized, they will be fed into a tensorflow model of an LSTM (Long/Short Term Memory) RNN (Recurrent Neural Network), which is a neural network that takes advantage of time-series data to learn and classify. An LSTM will be used because of its overall effectiveness. 

If all goes well and the model is good, the model will then be used to try and classify text I will give it myself, just to see if any tweaking needs to be done. Once all the necessary tweaking is done and I have a model with a good accuracy, I'll then proceed to use it for some other cool apps.

One interesting app could be to make a chatbot that would predict the mood of the user based on the words given to it, potentially providing for a more meaningful and in-depth conversation. One problem I also want to solve is the problem of loneliness and urgency in those who are mentally ill and suicidal, like myself. When someone is in 'crisis,' meaning they are emotionally overwhelmed or 'triggered' (which was originally a term to describe when one is in an intense and emotionally painful state of mind that they cannot get out of themselves), it's not always easy to communicate ones thoughts to other people given the frames of mind are in contrast with eachother.
What this project will try to solve is the problem of understanding someone who is in crisis by feeding suicidal posts from Reddit, and learning common themes in the text that a person might not be able to notice. Early identification is extremely important in stopping a further downward spiral of suicidal and self-deprecating thoughts. I could also implement sentiment analysis to track the progression of suicidal posts, and then train a model based on positive progressions of suicidal posts to then generate supportive text to a user's suicidal input.

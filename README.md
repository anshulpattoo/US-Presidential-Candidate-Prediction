# US-Presidential-Candidate-Prediction
The primary question our analytics attempts to answer is this: What kind of pattern in the language of a speech
predicts whether a candidate will win or lose an election? There are three specific goals we address while answering
this question. Our goals are:

1. To make the most accurate prediction of the likelihood of winning a US presidential election based on the
words used in speeches.
  a. Although the wording of this goal suggests a regression problem, this is a classification problem.
  Each row represents a campaign speech, and each column shows the frequency of a particular
  word in such speech. Thus, our model tells whether the speech it is given is associated with a
  winning campaign or a losing one. We rigorously assess the performances of various predictive
  models, and subsequently define a final model of choice.

2. To find some of the words that might be associated with winning speeches, and to see if they collectively
suggest any strategies for good speeches.
  a. Once we build a strong model that knows how to separate a winning speech from a losing one, we
  rank the attributes in terms of their predictive power, as this lets us see which words might be
  predictive of winning. We then discuss whether such words show any possible strategies for good
  speeches.

3. To see if deceptive language has any role in winning US presidential elections.

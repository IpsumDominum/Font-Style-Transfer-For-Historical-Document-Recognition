library(tidyverse)
library(qdapDictionaries)

### This R file runs the Viterbi algorithm on a subset of the King James bible 
### across different created emissions matrices, and compares the results.

### This code creates the transition matrix p
# read in text- currently bible.txt, i could make function if necessary
bible <- readLines("bible.txt")

# split each word
bible.words <- strsplit(bible, " ")
bible.words <- do.call(c, bible.words)

# create empty transition matrix, prior distribution, last letter counters
p <- matrix(0, nrow = 26, ncol = 26)
prior <- 1:26
last_letter <- 1:26

# for changing from letters -> numbers
myLetters <- letters[1:26]

# for each word
for (i in 1:length(bible.words)){
  word <- unlist(strsplit(bible.words[i], split = "")) #split into each character
  index <- match(word, myLetters) #make each character a number
  for (j in 1:length(index)-1){ # for each letter, except the last one in a word
    p[index[j], index[j+1]] <- p[index[j], index[j+1]] + 1 #increment the count of letter transition
  }
  
  last_letter[index[length(index)]] <- last_letter[index[length(index)]] + 1 #increment frequency of each letter being the last letter
}


# for each letter, find the prior distribution, and transform into probabilities
for (i in 1:26){
  prior[i] <- sum(p[i,])
  p[i,] <- p[i,] / prior[i]
  
}

#incorporate last ketter into priors
prior <- prior + last_letter
prior <- prior / sum(prior)

constructEmissions <- function(pr_correct,adj){
  
  # function creates the emissions matrix, at the moment this is simply based off of keyboard adjacency, so will change when we know the results of our character recognition
  
  b <- matrix(0, nrow = 26, ncol = 26) # create empty matrix for emissions
  for (i in 1:26){ # for each letter
    adj.sum <- sum(adj[i,]) #sum the number of adjacent letters
    b[i,] <- adj[i,] / adj.sum * (1 - pr_correct) #equally distribute probabilities
    b[i,i] <- pr_correct # probability of being correct
  }
  return (b)
}

pr_correct=0.9;

# ugly, will be replaced when we know our own emissions matrix
adj <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0)

# reformat above into matrix
adjacent <- matrix(adj, nrow = 26, ncol = 26)

# construct emissions
b <- constructEmissions(0.9, adjacent);
b2 <- constructEmissions(0.9, adjacent)
b3 <- constructEmissions(0.7, adjacent)
b4 <- constructEmissions(0.6, adjacent)
b5 <- constructEmissions(0.5, adjacent)

HMM <- function(p, prior, b, y){
  
  #function implements viterbi, finds most likely sequence of states
  
  n <- length(y) # length of word
  m <- length(prior) # length of prior, will always be 26
  gamma <- matrix(0, nrow = m, ncol = n)
  phi <- matrix(0, nrow = m, ncol = n)
  
  
  for (i in 1:26){
    gamma[i, 1] <- prior[i] * b[i, y[1]] # gamma value for time step 1
  }
  
  max.j <- 1:26 # array that will house the values of the expression 'gamma(j,t-1) * p(j,k) * b(k,y(t))', to allow for determining the j which maximises said expression
  
  for (t in 2:n){
    for (k in 1:26){
      for (j in 1:26){
        max.j[j] <- max(gamma[j, t-1] * p[j,k] * b[k, y[t]]) # for each j value, determine Viterbi value
      }
      gamma[k,t] <- max(max.j) # find the maximum j, place in gamma
      phi[k,t] <- which.max(max.j) # find the loocation of the maximum j, place in phi
    }
  }
  
  best <- 0
  x <- 0  * (1:n)
  
  for (k in 1:26){
    if (best <= gamma[k,n]){
      best <- gamma[k,n]
      x[n] <- k
    }
  }
  
  
  for (i in (n-1):1){
    x[i] <- phi[x[i+1], i+1] # return the value for likely letter stored in phi
  }
  
  return (x)
}


# create small function to test if word in dictionary
is.word <- function (x) x %in% GradyAugmented


Viterbi <- function(word, b){
  # should add b as input to function
  
  # if word in dictionary, then finish
  if (is.word(word)){
    return (word)
  } else { #if word not in dictionary, do Viterbi
    
    # make characters numbers  
    characters <- unlist(strsplit(word, split = ""))
    index <- match(characters, letters[1:26])#make letter[i] be readable
    
    # do viterbi
    x <- HMM(p, prior, b, index)
    
    # reformat back into a word
    new.letters <- letters[x]
    new.word <- noquote(paste(new.letters, collapse = ""))
    
    # return the 'likely' word
    return (new.word)
  }
  
}

change.word <- function(word, emissions){
  
  # for each character in a word
  characters <- unlist(strsplit(word, split = ""))
  
  # find the corresponding number
  index <- match(characters, letters)
  
  # the probability of being correct atm is stored here
  # with different emissions matrix this will change, can just add to function
  # definition
  
  pr.correct <- emissions[1,1]
  
  # generate a random number for each character
  N <- length(index)
  correct.character <- runif(N)
  
  # for each character in the word
  for (i in 1:N){
    
    # find the possible mistakes for this letter
    possible.characters <- which((emissions[index[i],] > 0) & (emissions[index[i],] < pr.correct))
    
    # if we change this letter
    if (correct.character[i] > pr.correct){
      #we change it to a possible character, all with equal likelihood
      # this will also change, as they will not all be equal likelihood
      change.letter <- sample(possible.characters, 1)
      index[i] <- change.letter
    }
  }
  
  # return the new word
  new.letters <- letters[index]
  new.word <- noquote(paste(new.letters, collapse = ""))
  
  return (new.word)
  
}

# subset the bible words, just for less computation
bible.subset <- readLines("bible_subset.txt")

# find each individual word
subset.words <- strsplit(bible.subset, " ")
subset.words <- do.call(c, subset.words)

# apply the changing words function
changed <- unlist(lapply(subset.words, change.word, emissions = b))
changed2 <- unlist(lapply(subset.words, change.word, emissions = b2))
changed3 <- unlist(lapply(subset.words, change.word, emissions = b3))
changed4 <- unlist(lapply(subset.words, change.word, emissions = b4))
changed5 <- unlist(lapply(subset.words, change.word, emissions = b5))

# these contain the viterbi results of each.
fixed <- unlist(lapply(changed, Viterbi, b = b))
fixed2<- unlist(lapply(changed2, Viterbi, b = b2))
fixed3<- unlist(lapply(changed3, Viterbi, b = b3))
fixed4<- unlist(lapply(changed4, Viterbi, b = b4))
fixed5<- unlist(lapply(changed5, Viterbi, b = b5))

# the original characters in every word
original.characters <- unlist(strsplit(subset.words, split = ""))

viterbi.results <- function(changed, fixed){
  
  cat("The results are:\n")
  
  changed.characters <- unlist(strsplit(changed, split = ""))
  fixed.characters <- unlist(strsplit(fixed, split = ""))
  
  cat("The old correct word percentage was:", 100 * mean(changed == subset.words), "\n")
  cat("The new correct word percentage was:", 100 * mean(fixed == subset.words), "\n")
  
  cat("The old correct character percentage was:", 100 * mean(changed.characters == original.characters), "\n")
  cat("The new correct character percentage was:", 100 * mean(fixed.characters == original.characters), "\n")
  
  tp <- mean((original.characters == changed.characters) & (original.characters == fixed.characters) & (fixed.characters == changed.characters))
  
  tn <- mean((original.characters != changed.characters) & (fixed.characters != changed.characters))
  
  fp <- mean((original.characters != changed.characters) & (original.characters != fixed.characters) & (fixed.characters == changed.characters))
  
  fn <- mean((original.characters == changed.characters) & (original.characters != fixed.characters) & (fixed.characters != changed.characters))
  
  cat("The true positive percentage is:", 100 * tp, "\n")
  cat("The true negative percentage is:", 100 * tn, "\n")
  cat("The false positive percentage is:", 100 * fp, "\n") 
  cat("The false negative percentage is:", 100 * fn, "\n")  
  
  # true positive rate
  tpr <- tp / (tp + fn)
  
  # false negative rate
  fnr <- fn / (tp + fn)
  
  # false positive rate
  fpr <- fp / (fp + tn)
  
  # true negative rate
  tnr <- tn / (fp + tn)
  
  cat("The true positive rate is:", tpr, "\n")
  cat("The true negative rate is:", tnr, "\n")
  cat("The false positive rate is:", fpr, "\n") 
  cat("The false negative rate is:", fnr, "\n")
  
  cat("\n")
}

viterbi.results(changed, fixed)
viterbi.results(changed2, fixed2)
viterbi.results(changed3, fixed3)
viterbi.results(changed4, fixed4)
viterbi.results(changed5, fixed5)

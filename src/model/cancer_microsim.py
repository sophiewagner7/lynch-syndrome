from src.model.markov import MarkovModel

# MarkovModel: D[t,s] gives us distribution at time t
# MarkovModel: I[t,s] gives us incidence at time t


# We need to take cancers once detected, and apply age-specific death rates? also to apply costs
def redo_cancer_deaths(model: MarkovModel):
    I = model.I
    I_yearly = I

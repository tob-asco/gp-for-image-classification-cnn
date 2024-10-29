from tqdm import tqdm
from torch.utils.data import Subset
from IPython.display import clear_output
from torch.utils.data import DataLoader
import torch
from torchmetrics import functional
import time
import torch.nn.functional as F
import individuals as inds
import random

def train_model_one_batch(ind: inds.NN_individual,       # <- model to be trained in-place
                          X, y,                     # <- train batch, e.g. X.shape = [32, 1, 28, 28]
                          ) -> tuple[float, float]: # -> return (train_loss, time that training took)
  train_loss = 0
  start_time = time.perf_counter() # Start timing
  ind.train()
  X, y = X.to(ind.device), y.to(ind.device)
  y_pred = ind(X)
  if inds.NN_dna.loss_fn_dict[ind.dna.loss_fn_gene]['1hot']: # <- does the loss_fn need one-hot encoding?
    loss = ind.loss_fn(y_pred, F.one_hot(y, num_classes=ind.CLASSIFICATION_CATEGORIES_COUNT).float())
  else:
    loss = ind.loss_fn(y_pred, y)
  train_loss += loss
  ind.optimizer.zero_grad()
  loss.backward()
  ind.optimizer.step()
  end_time = time.perf_counter() # Stop timing
  return (train_loss, end_time-start_time)

def test_model(ind: inds.NN_individual,            # <- model to be tested
               test_dl,                       # <- test dataloader (= multiple batches)
               ) -> tuple[float, float]:      # -> return (loss_total, acc_total)
  loss_total, acc_total = 0, 0
  ind.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dl):
      X, y = X.to(ind.device), y.to(ind.device)
      preds = ind(X)
      if inds.NN_dna.loss_fn_dict[ind.dna.loss_fn_gene]['1hot']: # <- does the loss_fn need one-hot encoding?
        loss_batch = ind.loss_fn(preds, F.one_hot(y, num_classes=ind.CLASSIFICATION_CATEGORIES_COUNT).float())
      else:
        loss_batch = ind.loss_fn(preds, y)
      loss_total += loss_batch
      acc_batch = functional.accuracy(preds, y, task="multiclass", num_classes=ind.CLASSIFICATION_CATEGORIES_COUNT)
      acc_total += acc_batch

    loss_total /= len(test_dl)
    acc_total /= len(test_dl)
  return (loss_total, acc_total)

def train_model_one_batch(ind: inds.NN_individual,       # <- model to be trained in-place
                          X, y,                     # <- train batch, e.g. X.shape = [32, 1, 28, 28]
                          ) -> tuple[float, float]: # -> return (train_loss, time that training took)
  train_loss = 0
  start_time = time.perf_counter() # Start timing
  ind.train()
  X, y = X.to(ind.device), y.to(ind.device)
  y_pred = ind(X)
  if inds.NN_dna.loss_fn_dict[ind.dna.loss_fn_gene]['1hot']: # <- does the loss_fn need one-hot encoding?
    loss = ind.loss_fn(y_pred, F.one_hot(y, num_classes=ind.CLASSIFICATION_CATEGORIES_COUNT).float())
  else:
    loss = ind.loss_fn(y_pred, y)
  train_loss += loss
  ind.optimizer.zero_grad()
  loss.backward()
  ind.optimizer.step()
  end_time = time.perf_counter() # Stop timing
  return (train_loss, end_time-start_time)

def test_model(ind: inds.NN_individual,            # <- model to be tested
               test_dl,                       # <- test dataloader (= multiple batches)
               ) -> tuple[float, float]:      # -> return (loss_total, acc_total)
  loss_total, acc_total = 0, 0
  ind.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dl):
      X, y = X.to(ind.device), y.to(ind.device)
      preds = ind(X)
      if inds.NN_dna.loss_fn_dict[ind.dna.loss_fn_gene]['1hot']: # <- does the loss_fn need one-hot encoding?
        loss_batch = ind.loss_fn(preds, F.one_hot(y, num_classes=ind.CLASSIFICATION_CATEGORIES_COUNT).float())
      else:
        loss_batch = ind.loss_fn(preds, y)
      loss_total += loss_batch
      acc_batch = functional.accuracy(preds, y, task="multiclass", num_classes=ind.CLASSIFICATION_CATEGORIES_COUNT)
      acc_total += acc_batch

    loss_total /= len(test_dl)
    acc_total /= len(test_dl)
  return (loss_total, acc_total)

def train_and_evaluate_gen(pop: inds.NN_population,
                           train_dl,  # <- train dataloader
                           test_dl,   # <- test dataloader
                           testing_interval = 300,      # <- after how many batches should we test an individual
                           testing_data_fraction = 1.0, # <- amount of test_dl to be used (1=100% takes a lot of time)
                           training_data_fraction = 1.0,# <- amount of train_dl to be used (1=100% takes a lot of time)
                           epochs = 5,
                           live_plot = True,
                           only_last_plot = False,
                           no_plot = False):
  start_time = time.perf_counter() # Start timing

  # prepare the reduced testing data loader
  random_batch_indices = random.sample(range(len(test_dl)), int(len(test_dl) * testing_data_fraction)) # random indices (without replacement)
  test_subset = Subset(test_dl.dataset, random_batch_indices)
  test_subset_dl = DataLoader(test_subset, batch_size=test_dl.batch_size, shuffle=False, num_workers=test_dl.num_workers)

  # prepare the reduced training data loader ()
  first_indices = list(range(int(len(train_dl)*train_dl.batch_size * training_data_fraction)))
  train_subset = Subset(train_dl.dataset, first_indices)
  train_subset_dl = DataLoader(train_subset, batch_size=train_dl.batch_size, shuffle=False, num_workers=train_dl.num_workers)

  # re-initialize pop's fitness values:
  for ind in pop:
    ind.acc, ind.running_acc = 0, 0
    ind.train_losses, ind.test_losses, ind.accs = {}, {}, {}
  
  # train each individual "simultaneously" by making the epoch-loop the outer one
  for epoch in range(epochs):
    print(f"*** Commencing epoch {epoch+1} / {epochs} for {len(pop)} individuals, one line each. ***")
    for i in range(len(pop)):
      p_bar = tqdm(enumerate(train_subset_dl))
      for batch, (X, y) in p_bar:
        # train the model (update the weights and biases of the NN pop[i])
        pop[i].train_losses[f"e_{epoch}@b_{batch}"], elapsed_batch_training_time = train_model_one_batch(pop[i], X=X, y=y)
        pop[i].elapsed_training_time += elapsed_batch_training_time
        if batch % testing_interval == 0: 
          # test the model and store the results
          pop[i].test_losses[f"e_{epoch}@b_{batch}"], pop[i].accs[f"e_{epoch}@b_{batch}"] = test_model(pop[i], test_dl=test_subset_dl)
          p_bar.set_description(f"{i+1}. {pop[i].name} [{pop[i].dna.toString()}] {pop[i].accs[f"e_{epoch}@b_{batch}"]*100:.1f}%") # update the progress bar to display the current accuracy
          if batch != 0: # don't use the start/benchmark test as this depends mostly on luck of weight initialization
            pop[i].running_acc += pop[i].accs[f"e_{epoch}@b_{batch}"] / pop[i].elapsed_training_time
          if live_plot and not only_last_plot and not no_plot:
            clear_output(wait=True)
            pop.plot_accs(time.perf_counter() - start_time)
      pop[i].test_losses[f"e_{epoch}@end"], pop[i].accs[f"e_{epoch}@end"] = test_model(pop[i], test_dl=test_dl) # latest precise values
      pop[i].acc = pop[i].accs[f"e_{epoch}@end"] # store the very last known accuracy
      if not live_plot and not only_last_plot and not no_plot:
        clear_output(wait=True)
        pop.plot_accs(time.perf_counter() - start_time)
    # here we could select directly, i.e. before the whole train_dl over max_epochs no. of iterations has been trained

  if not no_plot and not only_last_plot:
    clear_output(wait=True)
    pop.plot_accs(time.perf_counter() - start_time)
  elif not no_plot and only_last_plot:
    pop.plot_accs(time.perf_counter() - start_time)
  else:  
    print(f"This took {time.perf_counter() - start_time:.2f}s.")
      
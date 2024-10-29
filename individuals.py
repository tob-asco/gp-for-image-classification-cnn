import torch
from torch import nn
import re
import math
import matplotlib.pyplot as plt

# Calculate the image size after a layer has been applied assume all operations to be x/y symmetric
def output_size(h_in, kernel_size, stride, padding):
    h_out = (h_in + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    #w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return h_out # = w_out

# class for the genetic information of one 2d block
class Gene_2d_block:
    def __init__(self,
                 input_image_size: int,
                 out_channels: int,
                 conv_kernel_size: int = 3,
                 conv_stride: int = 1,
                 conv_padding: int = 1,
                 pool_kernel_size: int = 2,
                 pool_stride: int = 2,
                 pool_padding: int = 0):
        self.input_image_size = input_image_size
        self.out_channels = out_channels
        self.in_channels = None
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        self.after_conv_image_size = output_size(input_image_size, conv_kernel_size, conv_stride, conv_padding)
        self.output_image_size = output_size(self.after_conv_image_size, pool_kernel_size, pool_stride, pool_padding)

    def toString(self, tab_count: int = 0):
        indentation = ""
        for tab in range(tab_count): indentation += f"\t"
        return f"{indentation}out_channels = {self.out_channels} <-- ({self.input_image_size} x {self.input_image_size})\n"+\
        f"{indentation}conv_2d (kernel, stride, padding) =\t({self.conv_kernel_size}, {self.conv_stride}, {self.conv_padding}) --> ({self.after_conv_image_size} x {self.after_conv_image_size})\n"+\
        f"{indentation}max_pool_2d (kernel, stride, padding) =\t({self.pool_kernel_size}, {self.pool_stride}, {self.pool_padding}) --> ({self.output_image_size} x {self.output_image_size})"

''' A helper that gets an otpimizer_string where the given chars need to appear and be followed by a signed integer.
    E.g. optimizer_string="SGD@λ3μ-2χ0" and chars=['λ','χ','μ'] will produce the list [3.0, 0.0, -2.0]'''
def parse_optimizer_string(optimizer_string, chars = ['λ','μ']) -> list[float]:
    exponents = []
    for char in chars:
        match = re.search(fr'{char}([-+]?\d+)', optimizer_string) # Use regex to find the values following the char
        if match: exponents.append(float(match.group(1))) # Extract and convert these values to floats
        else: raise ValueError(f"Could not find '{char}' in the string.")
    return exponents

# class containing various genes (with their vanilla values) and the G2P mappings
class NN_dna():
    def __init__(self,
                 blocks_2d: list[Gene_2d_block] = [],
                 optimizer: int = 0,
                 lr: float = .1,
                 loss_fn: int = 0,
                 ) -> None:
        self.blocks_2d_gene = blocks_2d
        self.optimizer_gene = optimizer
        self.lr = lr
        self.loss_fn_gene = loss_fn
    
    #### Genotype To Phenotype Mappings (G2P) ####
    def blocks_2d_G2P(self, COLOUR_CHANNEL_COUNT: int):
        ''' build 'n return the full sequential from the gene information (genes_2d_block) 
            the first 2d_block needs to have as many in_channels as there are colour channels
            the others need to have as in_channels the number of out_channels from the previous block
            there's a nn.Module (Lazy*) that automatically infers the number of in_channels - not used here '''
        blocks_2d = nn.Sequential()
        for i in range(len(self.blocks_2d_gene)):
            if i == 0:
                in_channels = COLOUR_CHANNEL_COUNT
            else:
                in_channels = self.blocks_2d_gene[i-1].out_channels
            blocks_2d.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                    out_channels=self.blocks_2d_gene[i].out_channels,
                    kernel_size=self.blocks_2d_gene[i].conv_kernel_size,
                    stride=self.blocks_2d_gene[i].conv_stride,
                    padding=self.blocks_2d_gene[i].conv_padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=self.blocks_2d_gene[i].pool_kernel_size,
                    stride=self.blocks_2d_gene[i].pool_stride,
                    padding=self.blocks_2d_gene[i].pool_padding)))
        return blocks_2d
        
    optimizer_dict = {0: "SGD@λ0μ0", 1: "SGD@λ0μ1", 2: "SGD@λ0μ-1", 3: "SGD@λ1μ0", 4: "SGD@λ1μ1", 5: "SGD@λ1μ-1", 6: "SGD@λ-1μ0", 7: "SGD@λ-1μ1", 8: "SGD@λ-1μ-1", 9: "adam"}
    def optimizer_G2P(self, model_parameters):
        if self.optimizer_dict[self.optimizer_gene].lower().startswith("sgd"): 
            exps = parse_optimizer_string(self.optimizer_dict[self.optimizer_gene],chars=['λ','μ']) # find the exponents to use from the name I chose in optimizer_dict
            return torch.optim.SGD(model_parameters, lr=self.lr, weight_decay=math.exp(exps[0]), momentum=math.exp(exps[1])) # use Euler's e as base for the exponents
        if self.optimizer_gene == 9: return torch.optim.Adam(model_parameters, lr=self.lr)
        raise ValueError(f"'{self.optimizer_gene}' is not a gene for which we have an optimizer encoded.")
    
    # a loss function needs to come with information of how targets need to be encoded, either y = 2 (1hot=False) or y = [0,0,1,0,...] (1hot=True)
    loss_fn_dict = {0: {'name': "CE", '1hot': False}, 1: {'name': "L1", '1hot': True}, 2: {'name': "Hub@δ=.1", '1hot': True}, 3: {'name': "Hub@δ=1", '1hot': True}, 4: {'name': "Hub@δ=10", '1hot': True}}
    def loss_fn_G2P(self):
        if self.loss_fn_gene == 0: return nn.CrossEntropyLoss()
        ''' The following loss functions should only be used with one-hot encoded targets.
            This means that instead of e.g. y = 3, we need y = [0,0,0,1,0,0,0,0,0,0] if CATEGORIES_COUNT = 10.
            So all the (wrong) predictions - i.e. x_0,x_1,x_2,x_4,x_5,... - are ignored in the loss calculation.
            This is why usually CrossEn is used, according to ChatGPT'''
        if self.loss_fn_gene == 1: return nn.L1Loss()
        if self.loss_fn_gene == 2: return nn.HuberLoss(delta=.1)
        if self.loss_fn_gene == 3: return nn.HuberLoss(delta=1) # same as SmoothL1Loss
        if self.loss_fn_gene == 4: return nn.HuberLoss(delta=10)
        raise ValueError(f"'{self.loss_fn_gene}' is not a gene for which we have a loss function encoded.")

    def toString(self):
        s = "("
        for i, block in enumerate(self.blocks_2d_gene):
            s += f"{block.out_channels},"                           # e.g. (1,7,
        if len(s)>1: s = s[:-1]                                     # e.g. (1,7
        s += f") {self.optimizer_dict[self.optimizer_gene]}"        # e.g. (1,7) SGD
        s += f" {self.loss_fn_dict[self.loss_fn_gene]['name']}"     # e.g. (1,7) SGD CrossEn
        s += f" | {self.lr:.2g}"                                    # e.g. (1,7) SGD CrossEn | 1.1
        return s

''' class for image classification individuals
    Essentially, it converts NN_dna (provided to __init__)
    into a working NN with a forward method
'''
class NN_individual(nn.Module):
    def __init__(self,
                 COLOUR_CHANNEL_COUNT: int,
                 CLASSIFICATION_CATEGORIES_COUNT: int,
                 dna: NN_dna = NN_dna(),    # <- contains all genes
                 name: str = "nn0",         # <- name unique within a population/generation
                 device = "cpu"):
        super().__init__()
        self.COLOUR_CHANNEL_COUNT = COLOUR_CHANNEL_COUNT
        self.CLASSIFICATION_CATEGORIES_COUNT = CLASSIFICATION_CATEGORIES_COUNT
        self.dna = dna
        self.blocks_2d = dna.blocks_2d_G2P(COLOUR_CHANNEL_COUNT)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # default is: start_dim = 1
        self.lazyLin = nn.LazyLinear(out_features = CLASSIFICATION_CATEGORIES_COUNT) # automatically infers the number of channels
        self.name = name
        self.lr = dna.lr
        self.optimizer = dna.optimizer_G2P(self.parameters())
        self.loss_fn = dna.loss_fn_G2P()
        self.to(device)
        self.device = device

        self.acc = 0
        self.running_acc = 0
        self.train_losses = {}
        self.test_losses = {}
        self.accs = {}
        self.elapsed_training_time = 0
        
    def forward(self, x):
        for i in range(len(self.blocks_2d)):
            x = self.blocks_2d[i](x)
        x = self.flatten(x)
        x = self.lazyLin(x)
        return x


# created by Chat
class NN_population:
    ### Magic Methods ###
    def __init__(self, individuals: list[NN_individual]): self.individuals = individuals
    def __getitem__(self, index): return self.individuals[index]  # magic pop[i] access
    def __len__(self): return len(self.individuals)  # magic len(pop)
    def __setitem__(self, index, value): self.individuals[index] = value  # magic pop[i] = value
    def __iter__(self): return iter(self.individuals)  # magic for-iterations
    
    def plot_accs(self, elapsed_time = 0):
        plt.figure(figsize=(15, 6))  # Set the figure size
        for ind in sorted(self.individuals, key=lambda ind: ind.running_acc, reverse=True):
            x = list(ind.accs.keys())  # Extract the epoch/batch labels (x-axis)
            y = [float(val.cpu().item()*100) for val in ind.accs.values()]  # Convert tensors to floats
            # Plot each individual's accuracies
            plt.plot(x, y, marker='o', linestyle='-', label=f"{ind.name} [{ind.dna.toString()}] ({ind.running_acc:.2f} within {ind.elapsed_training_time:.1f}s) {ind.acc*100:.1f}%")
        plt.xlabel('Epoch@Batch')  # Label for the x-axis
        plt.ylabel('Accuracy [%]')     # Label for the y-axis
        extra_title = "" if elapsed_time == 0 else f" (took {elapsed_time:.2f}s)"
        plt.title('Accuracy per Epoch and Batch' + extra_title)  # Title of the plot
        plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
        plt.grid(True)  # Show grid
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # legend on the right
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot area size to leave space for the legend
        plt.show()

print(NN_population([NN_individual(1,10)]))
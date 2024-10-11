import random, copy, torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
pd.set_option('display.expand_frame_repr', False)
import evaluation as eval
import individuals as inds
import constants

def create_random_2d_block(input_image_size, max_kernel_size: int) -> inds.Gene_2d_block:
    conv_kernel_size=min(random.randint(1,min(input_image_size, max_kernel_size)), random.randint(1,min(input_image_size, max_kernel_size))) # kernel must be smaller than image size!
    conv_stride=random.randint(1,conv_kernel_size)
    conv_padding=random.randint(0,conv_kernel_size//2) # PyTorch: "pad should be at most half of effective kernel size"
    after_conv_i_s = inds.output_size(input_image_size, conv_kernel_size, conv_stride, conv_padding)
    pool_kernel_size=min(random.randint(1,min(after_conv_i_s, max_kernel_size)), random.randint(1,min(after_conv_i_s, max_kernel_size)), random.randint(1,min(after_conv_i_s, max_kernel_size)))
    if conv_kernel_size == 0 or pool_kernel_size == 0:
        print("Exception! A kernel size is 0, which is not allowed.")
        print(f"input_image_size {input_image_size}, max_kernel_size {max_kernel_size}, conv_kernel_size {conv_kernel_size}, conv_stride {conv_stride}, conv_padding {conv_padding}, after_conv_image_size {after_conv_i_s}, pool_kernel_size {pool_kernel_size}")
    pool_stride=max(random.randint(1,pool_kernel_size), random.randint(1,pool_kernel_size))
    pool_padding=random.randint(0,pool_kernel_size//2) # PyTorch: "pad should be at most half of effective kernel size"
    return inds.Gene_2d_block(
        input_image_size=input_image_size,
        out_channels=random.randint(3,15), # not fine-tuned
        conv_kernel_size=conv_kernel_size,
        conv_padding=conv_padding,
        conv_stride=conv_stride,
        pool_kernel_size=pool_kernel_size,
        pool_padding=pool_padding,
        pool_stride=pool_stride
    )

def update_and_check_2d_block_stack(gene_2d_blocks: list[inds.Gene_2d_block], PC: constants.problem_constants):
    protocol = ""
    if len(gene_2d_blocks) == 0: return "no block in the stack"
    if gene_2d_blocks[0].input_image_size != PC.height:
        gene_2d_blocks[0].input_image_size != PC.height
        protocol += f"block[0]'s input_image_size was set to IMAGE_HEIGHT ({PC.height}), " 
    for i, block in enumerate(gene_2d_blocks):
        if i > 0 and block.input_image_size != gene_2d_blocks[i-1].output_image_size:
            block.input_image_size = gene_2d_blocks[i-1].output_image_size
            protocol += f"block[{i}]'s input_image_size was set to block[{i-1}]'s output_image_size ({block.input_image_size}), "
            block.after_conv_image_size = inds.output_size(block.input_image_size, block.conv_kernel_size, block.conv_stride, block.conv_padding) # update the effective image size after convolution
        if block.input_image_size < block.conv_kernel_size:
            block.conv_kernel_size = block.input_image_size
            protocol += f"block[{i}]'s conv_kernel_size was decreased to input_image_size ({block.conv_kernel_size}), "
        if block.conv_padding > block.conv_kernel_size // 2:
            block.conv_padding = block.conv_kernel_size // 2
            protocol += f"block[{i}]'s conv_padding was decreased to conv_kernel_size//2 ({block.conv_kernel_size // 2}), "
        if block.after_conv_image_size < block.pool_kernel_size:
            block.pool_kernel_size = block.after_conv_image_size
            protocol += f"block[{i}]'s pool_kernel_size was decreased to after_conv_size ({block.pool_kernel_size}), "
        if block.pool_padding > block.pool_kernel_size // 2:
            block.pool_padding = block.pool_kernel_size // 2
            protocol += f"block[{i}]'s pool_padding was decreased to pool_kernel_size//2 ({block.pool_kernel_size // 2}), "
    return protocol
    
def create_random_population(pop_size: int,
                             PC: constants.problem_constants,
                             max_2d_block_count: int = 3, 
                             max_kernel_size: int = 11,
                             name_prefix="nn",
                             device="cpu",
                             print_summary: bool = True) -> inds.NN_population:
    population = []
    for i in range(pop_size):
        genes_2d_block = []
        input_image_size = PC.height # = IMAGE_WIDTH (assumed)
        name=name_prefix+str(i)
        if print_summary: print(f"Individual '{name}' <-- ({input_image_size} x {input_image_size})")
        for j in range(random.randint(1, max_2d_block_count)):
            if print_summary: print(f"\tBlock {j}")
            # create a random conv-pool block and store the corresponding new input_image_size for the block thereafter
            block = create_random_2d_block(input_image_size, max_kernel_size)
            input_image_size = block.output_image_size
            genes_2d_block.append(block)
            if print_summary: print(f"{block.toString(tab_count=2)} --> ({input_image_size} x {input_image_size})")
        dna = inds.NN_dna(blocks_2d=genes_2d_block,
                     loss_fn=random.randrange(len(inds.NN_dna.loss_fn_dict)),
                     optimizer=random.randrange(len(inds.NN_dna.optimizer_dict)),
                    # here you can change the remaining hyperparameters
                    )
        population.append(inds.NN_individual(PC=PC, dna=dna, name=name, device=device))
    return inds.NN_population(population)

class Mutation():
    def __init__(self, PC: constants.problem_constants, MAX_KERNEL_SIZE = 11):
        self.PC = PC
        self.R = random.random() # <- radicality of this mutation instance
        self.MAX_KERNEL_SIZE = MAX_KERNEL_SIZE

    ### factors of impact ###
    impact_lr_factor_1_1=.3
    impact_lr_factor_10=3
    impact_add_neuron_to_2d_block=1
    impact_delete_neuron_from_2d_block=1
    impact_add_2d_block=5
    impact_delete_2d_block=5
    impact_increase_kernel=.8
    impact_decrease_kernel=.8
    impact_change_optimizer=5
    impact_change_loss_fn=5

    ### possible operations (p_raw = 0 will happen most likely, p_raw = 1 least likely, p_raw = -1 definitely) ###
    def lr_factor_1_1(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_lr_factor_1_1 < self.R:
            if random.random() > .5:
                dna.lr *= 1.1
                return "multiplied lr by 1.1"
            else:
                dna.lr /= 1.1
                return "divided lr by 1.1"
    def lr_factor_10(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_lr_factor_10 < self.R:
            if random.random() > .5:
                dna.lr *= 10
                return "multiplied lr by 10"
            else:
                dna.lr /= 10
                return "divided lr by 10"
    def add_neuron_to_2d_block(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_add_neuron_to_2d_block < self.R:
            if len(dna.blocks_2d_gene) == 0: return None
            layer_nr = random.randrange(0, len(dna.blocks_2d_gene))
            dna.blocks_2d_gene[layer_nr].out_channels += 1
            return f"added neuron to 2d block no. {layer_nr}"
    def delete_neuron_from_2d_block(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_delete_neuron_from_2d_block < self.R:
            if len(dna.blocks_2d_gene) == 0: return None
            layer_nr = random.randrange(0, len(dna.blocks_2d_gene))
            if dna.blocks_2d_gene[layer_nr].out_channels > 1: dna.blocks_2d_gene[layer_nr].out_channels -= 1
            return f"deleted neuron from 2d block no. {layer_nr}"
    def add_2d_block(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_add_2d_block < self.R:
            layer_nr = random.randrange(0, len(dna.blocks_2d_gene) + 1)
            input_image_size = self.PC.width if layer_nr == 0 else dna.blocks_2d_gene[layer_nr-1].output_image_size
            dna.blocks_2d_gene.insert(layer_nr, create_random_2d_block(input_image_size, self.MAX_KERNEL_SIZE))
            protocol = update_and_check_2d_block_stack(dna.blocks_2d_gene, PC=self.PC) # check whether this insertion "killed" the entity (and if yes: repair it)
            return f"added block @ {layer_nr} ({protocol})"
    def delete_2d_block(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_delete_2d_block < self.R:
            if len(dna.blocks_2d_gene) == 0: return None
            layer_nr = random.randrange(0, len(dna.blocks_2d_gene))
            dna.blocks_2d_gene.pop(layer_nr)
            protocol = update_and_check_2d_block_stack(dna.blocks_2d_gene, PC=self.PC) # check whether this deletion "killed" the entity (and if yes: repair it)
            return f"deleted 2d block at {layer_nr} ({protocol})"
    def increase_kernel(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_increase_kernel < self.R:
            if len(dna.blocks_2d_gene) == 0: return None
            layer_nr = random.randrange(0, len(dna.blocks_2d_gene))
            if random.random() > .5:
                if dna.blocks_2d_gene[layer_nr].conv_kernel_size < dna.blocks_2d_gene[layer_nr].input_image_size: # check whether the kernel may be increased
                    dna.blocks_2d_gene[layer_nr].conv_kernel_size += 1
                    return f"conv kernel += 1 of 2d block no. {layer_nr}"
            else:
                if dna.blocks_2d_gene[layer_nr].pool_kernel_size < dna.blocks_2d_gene[layer_nr].after_conv_image_size:
                    dna.blocks_2d_gene[layer_nr].pool_kernel_size += 1
                    return f"pool kernel +=1 of 2d block no. {layer_nr}"
    def decrease_kernel(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_decrease_kernel < self.R:
            if len(dna.blocks_2d_gene) == 0: return None
            layer_nr = random.randrange(0, len(dna.blocks_2d_gene))
            if random.random() > .5:
                if dna.blocks_2d_gene[layer_nr].conv_kernel_size > 1 and dna.blocks_2d_gene[layer_nr].conv_padding*2 < dna.blocks_2d_gene[layer_nr].conv_kernel_size:
                    dna.blocks_2d_gene[layer_nr].conv_kernel_size -= 1
                    return f"conv kernel -= 1 of 2d block no. {layer_nr}"
            else:
                if dna.blocks_2d_gene[layer_nr].pool_kernel_size > 1 and dna.blocks_2d_gene[layer_nr].pool_padding*2 < dna.blocks_2d_gene[layer_nr].pool_kernel_size:
                    dna.blocks_2d_gene[layer_nr].pool_kernel_size -= 1
                    return f"pool kernel -= 1 of 2d block no. {layer_nr}"
    def change_optimizer(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_change_optimizer < self.R:
            optimizer_index = random.randrange(0, len(inds.NN_dna.optimizer_dict))
            if optimizer_index == dna.optimizer_gene: return None
            dna.optimizer_gene = optimizer_index
            return f"changed optimizer to {inds.NN_dna.optimizer_dict[optimizer_index]}"
    def change_loss_fn(self, dna: inds.NN_dna, p_raw: float = -1):
        if p_raw * self.impact_change_loss_fn < self.R:
            loss_fn_index = random.randrange(0, len(inds.NN_dna.loss_fn_dict))
            if loss_fn_index == dna.loss_fn_gene: return None
            dna.loss_fn_gene = loss_fn_index
            return f"changed loss function to {inds.NN_dna.loss_fn_dict[loss_fn_index]['name']}"

def mutant_from_dna(dna_parent: inds.NN_dna, PC: constants.problem_constants, print_actions: bool = True, mutant_name = "NN", device = "cpu") -> inds.NN_individual:
    # clone the dna (to not change the parent's dna)
    dna_mutant = copy.deepcopy(dna_parent) 

    # create a test batch (to check wether the mutant is still "alive")
    test_batch = torch.rand(PC.batch_size, PC.channel_count, PC.width, PC.height).to(device)

    # create a mutation instance (this produces a radicality R)
    m = Mutation(PC=PC)
    if print_actions: print(f"Radicality in creation of '{mutant_name}': {m.R:.2g}")

    # Dynamically loop over all functions of the Mutation class
    for method_name in dir(m): # Loop through all attributes of the class
        if callable(getattr(m, method_name)) and not method_name.startswith("__"): # Filter to only functions (ignoring private methods and attributes)
            p_raw = random.random() # raw likelihood
            operation = getattr(m, method_name)
            mutation_healthy, deaths = False, 0
            while not mutation_healthy:
                dna_before_mutation = copy.deepcopy(dna_mutant) # clone the dna (to not change the parent's dna)
                effect = operation(dna_mutant, p_raw) # execute the operation
                mutant = inds.NN_individual(PC=PC, dna=dna_mutant, name=mutant_name, device=device)
                try: 
                    mutant(test_batch)
                    mutation_healthy = True
                except Exception as e:
                    if print_actions: print(f"\tLETHAL MUTATION ({effect}) (e: '{e}')")
                    dna_mutant = dna_before_mutation # restore to the state before lethal operation
                    deaths += 1
                    if deaths > 10: raise Exception("Too many deaths.")
            if effect != None and print_actions: print(f"- {effect}")
    return mutant

class GP_run():
    def __init__(self,
                 initial_pop,
                 train_dl,
                 test_dl,
                 PC: constants.problem_constants,
                 ini_pop_size = 10,
                 final_gen = 5,
                 max_2d_block_count = 5,
                 max_kernel_size = 11,
                 testing_interval_in_percent = 20, # 100 = test only once, 50 = test twice, ...
                 epochs = 1,
                 testing_data_fraction = 1,
                 train_data_fraction = .3,
                 kill_every_n_th = 2,
                 derivatives = ['a','b'],
                 device = "cpu"
                 ):
        self.initial_pop = initial_pop
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.PC = PC
        self.ini_pop_size = ini_pop_size
        self.final_gen = final_gen
        self.max_2d_block_count = max_2d_block_count
        self.max_kernel_size = max_kernel_size
        self.testing_interval_in_percent = testing_interval_in_percent
        self.epochs = epochs
        self.testing_data_fraction = testing_data_fraction
        self.train_data_fraction = train_data_fraction
        self.kill_every_n_th = kill_every_n_th
        self.derivatives = derivatives
        self.device = device
                 
        self.gens=[self.initial_pop]
        self.all_inds=[ind for ind in self.gens[0]]

    def resume(self, for_n_gens: int, mutation_verbosity = False, summary = True):
        gen_count_start = len(self.gens)
        while(True):
            if not (gen_count_start == len(self.gens) and gen_count_start > 1): # if the run has already been run (i.e. a former resume() call), don't do the following 3 lines
                print(f"***** Gen. {len(self.gens)} / {gen_count_start + for_n_gens} *****")
                eval.train_and_evaluate_gen(self.gens[-1], self.train_dl, self.test_dl, self.testing_interval_in_percent, self.testing_data_fraction, training_data_fraction=self.train_data_fraction, epochs=self.epochs, live_plot=False, only_last_plot=True, no_plot=True)
                for ind in self.gens[-1]: self.all_inds.append(ind)
            if len(self.gens)-gen_count_start >= for_n_gens: break # exit if we trained for desired number of gens
            survivors = sorted(self.gens[-1], key=lambda ind: ind.running_acc, reverse=True)[:len(self.gens[-1]) // self.kill_every_n_th] # pick the top 1/kill_every_n_th (w.r.t. running_acc)
            self.gens.append(inds.NN_population([mutant_from_dna(ind.dna, self.PC, mutant_name=ind.name+"."+v, print_actions=mutation_verbosity ,device=self.device) for ind in survivors for v in self.derivatives])) # mutate each survivor (as often as len(derivatives))

        if summary:
            print(f"******* SUMMARY *******")
            inds.NN_population(self.all_inds).plot_accs()

    def return_evolution_df(self, sort_by_running_acc=False):
        inds = sorted([self.gens[i][j] for i in range(len(self.gens)) for j in range(len(self.gens[i]))], key=lambda ind: ind.name) # all individuals of all generations, sorted by name
        rows = []
        for ind in inds:
            rows.append([ind.name, '['+ind.dna.toString()+']', ind.running_acc, ind.acc.item()*100, f"{ind.elapsed_training_time:.2f}s"])
        df = pd.DataFrame(rows, columns=['Name','DNA','running_acc','acc','training time'])
        if sort_by_running_acc: df = df.sort_values(by='running_acc',ascending=False)
        # Background colour
        cmap_bad_good = LinearSegmentedColormap.from_list("custom_map", ["darkred", "white", "green"])
        df_styled = df.style.background_gradient(cmap_bad_good).format({'running_acc':"{:.3g}",'acc':"{:.1f}"}).set_properties(**{'text-align': 'left'})

        return df_styled
    
    def pedegree_plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Store positions for each individual (to create branches)
        positions = {}  
        branch_lines = []
        colors = {}  # To assign a unique color for each initial individual
        initial_names = [ind.name for ind in self.gens[0]]  # Get the names of initial individuals

        # Generate a colormap with distinct colors for each initial individual
        color_map = plt.colormaps.get_cmap('tab10')  # Get a colormap with enough unique colors

        for i, name in enumerate(initial_names):
            colors[name] = color_map(i)

        # Plot each generation
        for i, gen in enumerate(self.gens):
            for ind in gen:
                name = ind.name
                # Find the root ancestor (initial individual) for this individual
                root_name = name.split('.')[0]

                # Determine parent (if exists)
                parent_name = name.rsplit('.', 1)[0] if '.' in name else None

                # Store the current generation and its value for the plot
                x = i  # Generation is the x-axis
                y = ind.acc.cpu()*100  # Value is the y-axis
                
                # Add the individual to the positions dict
                if name not in positions:
                    positions[name] = (x, y)

                # If the individual has a parent, create a branch line
                if parent_name and parent_name in positions:
                    parent_x, parent_y = positions[parent_name]
                    branch_lines.append([(parent_x, parent_y), (x, y), root_name])

                # Plot the individual as a point (using root_name color)
                ax.scatter(x, y, color=colors[root_name])

        # Draw the branch lines
        for branch in branch_lines:
            (x0, y0), (x1, y1), root_name = branch
            ax.plot([x0, x1], [y0, y1], color=colors[root_name])

        # Create a legend for the initial individuals
        handles = [plt.Line2D([0], [0], color=colors[name], lw=2, label=name) for name in initial_names]
        ax.legend(handles=handles, title='Lineage', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set labels
        ax.set_xlabel("Generations")
        ax.set_ylabel("Accuracy [%]")
        ax.set_title("NN Individual Evolution Across Generations")

        # Show the plot
        plt.show()

from dataclasses import dataclass, field
from typing import List as TypingList, Tuple
import math


# Define the constants
KB: float = 1.3806505E-23
H: float = 6.62606896E-34
KCAL_TO_EV: float = 0.0433641


# Define a structure to contain the details of a species
@dataclass
class Species:
    name: str                # The name of the species
    multiple: int            # The multiple of the species in the chemical equation
    list_index: int          # The index of the species in the list


# Define a structure to contain the details of a chemical reaction
@dataclass
class Reaction:
    reactants: TypingList[Species]                # Each species of the reactants
    products: TypingList[Species]                 # Each species of the products
    barrier: Tuple[float, float]                  # Forward and reverse barrier


# Define a structure to list the informations of all the species
@dataclass
class SpeciesList:
    name: str                                     # The name of the species
    concentration: float                          # The concentration of the species
    reaction_index: TypingList[int] = field(default_factory=list)       # The indices of the reactions that involves this species
    reaction_multiple: TypingList[float] = field(default_factory=list)   # The multiples of this species in those reactions


# Define a structure to contain the parameters for simulation
@dataclass
class Para:
    temperature: float
    dt: float
    max_step: int
    print_step: int



def split_multiple_species(multiple_species: str) -> Tuple[int, str]:
    index: int = 0
    contain_letter: bool = False
    for i, ch in enumerate(multiple_species):
        if not ch.isdigit():
            index = i
            contain_letter = True
            break

    # If the item contains some letters, split the multiple and the species
    if contain_letter:
        # If there is no digit in front of the item, the multiple is 1
        if index == 0:
            multiple = 1
            species = multiple_species
        # Else, split the multiple and the species
        else:
            try:
                multiple = int(multiple_species[0:index])
            except Exception:
                raise Exception(f"\n\n\n ERROR: There is some problem with the species '{multiple_species}' in the input file. \n\n\n")
            species = multiple_species[index:len(multiple_species)]
    # If the item is pure a number, print the error message
    else:
        raise Exception(f"\n\n\n ERROR: There is some problem with the species '{multiple_species}' in the input file. \n\n\n")

    return multiple, species



def read_a_chemical_reaction(line: str) -> Reaction:
    # Collect the words in the line
    words = line.split()
    index: int = 0

    # Obtain the reactants and reactant_multiples
    reactants: TypingList[Species] = []
    while True:
        if index < len(words):
            multiple, name = split_multiple_species(words[index])
            species = Species(
                name=name,
                multiple=multiple,
                list_index=0,
            )
            reactants.append(species)
            index += 1
        else:
            raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")

        if index < len(words):
            if words[index] == "+":
                index += 1
            elif words[index] == "->":
                index += 1
                break
            else:
                raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")
        else:
            raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")

    # Obtain the products, product_multiples and the forward_barrier
    products: TypingList[Species] = []
    forward_barrier: float = 0.0
    while True:
        if index < len(words):
            multiple, name = split_multiple_species(words[index])
            species = Species(
                name=name,
                multiple=multiple,
                list_index=0,
            )
            products.append(species)
            index += 1
        else:
            raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")

        if index < len(words):
            if words[index] == "+":
                index += 1
            else:
                if (words[index][0:1] == "(") and (words[index][len(words[index]) - 1:len(words[index])] == ","):
                    try:
                        forward_barrier = float(words[index][1:len(words[index]) - 1])
                    except Exception:
                        raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")
                    index += 1
                    break
        else:
            raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")

    # Obtain the reverse barrier
    reverse_barrier: float = 0.0
    if index < len(words):
        if words[index][len(words[index]) - 1:len(words[index])] == ")":
            try:
                reverse_barrier = float(words[index][0:len(words[index]) - 1])
            except Exception:
                raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")
    else:
        raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{line}'\n\n\n")

    # Return the chemical reaction
    return Reaction(
        reactants=reactants,
        products=products,
        barrier=(forward_barrier, reverse_barrier),
    )



def read_input(filename: str) -> Tuple[TypingList[Reaction], TypingList[SpeciesList], Para]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        raise Exception("\n\n\n ERROR: There is some problem in reading the input file. \n\n\n")

    lines_iter = iter(content.splitlines())     # Take the iterator for the lines of the file

    reactions: TypingList[Reaction] = []        # Contains all the reactions
    lists: TypingList[SpeciesList] = []         # Contains all the species
    temperature: float = 0.0
    dt: float = -1.0
    max_step: int = 0
    print_step: int = 0

    while True:
        try:
            value = next(lines_iter)
        except StopIteration:
            break

        words = value.split()                   # Collect the words in the line
        # If the line is empty or the annotation line (begin with //), ignore it; else, process it
        if (len(words) != 0) and (words[0] != "//"):
            # For chemical equation line
            if len(words) > 4:
                reactions.append(read_a_chemical_reaction(value))

            # For concentration line
            elif len(words) == 2:
                name = words[0]
                try:
                    concentration = float(words[1])
                except Exception:
                    raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")
                lists.append(SpeciesList(name=name, concentration=concentration))

            # For parameter line
            elif len(words) == 3:
                if (words[0] == "T") and (words[1] == "="):
                    try:
                        temperature = float(words[2])
                    except Exception:
                        raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")
                elif (words[0] == "dt") and (words[1] == "="):
                    try:
                        dt = float(words[2])
                    except Exception:
                        raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")
                elif (words[0] == "max_step") and (words[1] == "="):
                    try:
                        max_step = int(words[2])
                    except Exception:
                        raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")
                elif (words[0] == "print_step") and (words[1] == "="):
                    try:
                        print_step = int(words[2])
                    except Exception:
                        raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")
                else:
                    raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")

            # For illegal line
            else:
                raise Exception(f"\n\n\n ERROR: There is some problem with the following line in the input file.\n '{value}'\n\n\n")

    # Set the concentration of the uninitialized species to be zero
    for i in range(len(reactions)):
        for j in range(len(reactions[i].reactants)):
            exist = False
            for k in range(len(lists)):
                if reactions[i].reactants[j].name == lists[k].name:
                    exist = True
                    break
            if exist is False:
                name = reactions[i].reactants[j].name
                concentration = 0.0
                lists.append(SpeciesList(name=name, concentration=concentration))

        for j in range(len(reactions[i].products)):
            exist = False
            for k in range(len(lists)):
                if reactions[i].products[j].name == lists[k].name:
                    exist = True
                    break
            if exist is False:
                name = reactions[i].products[j].name
                concentration = 0.0
                lists.append(SpeciesList(name=name, concentration=concentration))

    # Obtain the list_index for the reactants and products in the reactions, along with the reaction_index and reaction_multiple in the list
    for i in range(len(reactions)):                    # For each reaction
        # Obtain the list_index for the reactants in reaction i, along with the reaction_index and reaction_multiple in the list
        for j in range(len(reactions[i].reactants)):  # For each reactant
            # Find out the list_index of reactant j
            for k in range(len(lists)):               # For each species in the lists
                if reactions[i].reactants[j].name == lists[k].name:
                    reactions[i].reactants[j].list_index = k
                    lists[k].reaction_index.append(i)
                    lists[k].reaction_multiple.append(reactions[i].reactants[j].multiple * (-1.0))
                    break

        # Obtain the list_index for the products in reaction i, along with the reaction_index and reaction_multiple in the list
        for j in range(len(reactions[i].products)):   # For each product
            # Find out the list_index of product j
            for k in range(len(lists)):               # For each species in the lists
                if reactions[i].products[j].name == lists[k].name:
                    reactions[i].products[j].list_index = k
                    lists[k].reaction_index.append(i)
                    lists[k].reaction_multiple.append(float(reactions[i].products[j].multiple))
                    break

    if temperature < 0.000001:
        raise Exception("\n\n\n ERROR: Hasn't define the temperature in the input file, or the temperature is smaller than 0.000001. \n\n\n")
    if dt < 0.0:
        raise Exception("\n\n\n ERROR: Hasn't define dt in the input file, or dt is smaller than 0.0. \n\n\n")
    if max_step == 0:
        raise Exception("\n\n\n ERROR: Hasn't define max_step in the input file, or max_step is 0. \n\n\n")
    if print_step == 0:
        raise Exception("\n\n\n ERROR: Hasn't define print_step in the input file, or print_step is 0. \n\n\n")

    para = Para(
        temperature=temperature,
        dt=dt,
        max_step=max_step,
        print_step=print_step,
    )

    return reactions, lists, para



def routine_reactor(reactions: TypingList[Reaction], lists: TypingList[SpeciesList], para: Para) -> None:
    # Get the parameters for the simulation
    rt: float = (8.31451 / 96485.3) * para.temperature
    a: float = (KB * para.temperature) / H

    # Calculate the reaction rate constants for all the reactions (0 for forward reaction, 1 for reverse reaction)
    rate_con: TypingList[Tuple[float, float]] = []
    for i in range(len(reactions)):
        rate_con.append((
            math.exp(-1.0 * reactions[i].barrier[0] * KCAL_TO_EV / rt) * a,
            math.exp(-1.0 * reactions[i].barrier[1] * KCAL_TO_EV / rt) * a,
        ))

    # Write the header to the concentration output file
    with open("concentrations", "w", encoding="utf-8") as concentration_output, open("reaction_rate", "w", encoding="utf-8") as reaction_rate_output:
        concentration_output.write("                          step                          time")
        for i in range(len(lists)):
            concentration_output.write(f"{lists[i].name:>30}")
        concentration_output.write("\n                             0             0.000000000000000")
        for i in range(len(lists)):
            concentration_output.write(f" {lists[i].concentration:29.15f}")

        # Write the header to the reaction rate output file
        reaction_rate_output.write("                          step                          time")
        for i in range(len(reactions)):
            reaction_rate_output.write(f" {'R' + str(i + 1):>30}")

        # Simulate the reactions iteratively
        rate: TypingList[float] = [0.0 for _ in range(len(reactions))]     # To restore the reaction rate of all the reactions
        t: float = 0.0
        for i in range(1, para.max_step + 1):
            # Calculate the reaction rates for all the reactions at step i
            for j in range(len(reactions)):                                 # For each reaction
                # Calculate the forward reaction rate for reaction j
                r0 = rate_con[j][0]                                         # Multiple the corresponding reaction rate constant
                for k in range(len(reactions[j].reactants)):                # For each reactant
                    # Multiple the concentration of reactant k
                    r0 *= lists[reactions[j].reactants[k].list_index].concentration ** reactions[j].reactants[k].multiple

                # Calculate the reverse reaction rate for reaction j
                r1 = rate_con[j][1]                                         # Multiple the corresponding reaction rate constant
                for k in range(len(reactions[j].products)):                 # For each product
                    # Multiple the concentration of product k
                    r1 *= lists[reactions[j].products[k].list_index].concentration ** reactions[j].products[k].multiple

                # Calculate the whole reaction rate for reaction j
                rate[j] = r0 - r1

            # Calculate the variation of all the species
            for j in range(len(lists)):                                     # For each species
                # Calculate the variation of species j
                dc = 0.0
                for k in range(len(lists[j].reaction_index)):               # For each reaction that involves species j
                    dc += rate[lists[j].reaction_index[k]] * lists[j].reaction_multiple[k]
                dc *= para.dt                                               # Multiply the time step

                # Update the variation of species j
                if lists[j].concentration + dc > 0.0:
                    lists[j].concentration += dc
                elif lists[j].concentration + dc > -0.000000000000001:
                    lists[j].concentration = 0.0
                else:
                    raise Exception(
                        f"\n\n\n ERROR: At step '{i}', the concentration of species '{lists[j].name}' is '{lists[j].concentration}', but its variation is '{dc}'. \n\n\n"
                    )

            # Output the concentration and the reaction rate
            t += para.dt
            if (i % para.print_step) == 0:
                # Output the concentration at current step
                concentration_output.write(f"\n{i:>30} {t:29.15f}")
                for j in range(len(lists)):
                    concentration_output.write(f" {lists[j].concentration:29.15f}")

                # Output the reaction rate at current step
                reaction_rate_output.write(f"\n{i:>30} {t:29.15f}")
                for j in range(len(rate)):
                    reaction_rate_output.write(f" {rate[j]:29.15f}")



def main() -> None:
    reactions, lists, para = read_input("input")
    routine_reactor(reactions, lists, para)


if __name__ == "__main__":
    main()

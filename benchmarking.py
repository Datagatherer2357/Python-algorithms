# Gareth Duffy GMIT- g00364693
# Computational Thinking with Algorithms Project
# Complete integrated benchmarking program

# Warm up of each algorithm before benchmarking:

# Insertion Sort [22]:

def insertionSort(alist):
   for index in range(1,len(alist)): # loop set from index [1] to end of array length

     currentvalue = alist[index] # Temporary variable set for current index array element
     position = index # Denotes the separation point between the two parts

     while position>0 and alist[position-1]>currentvalue: # While loop finds the insertion point/position within the sorted array
         alist[position]=alist[position-1]
         position = position-1 # Shifts the elements down to make room for the next element

     alist[position]=currentvalue # Assigning current value variable to new (sorted) position in alist array

alist = [7,6,2,5,3,1] # Unsorted array
insertionSort(alist) # Calling the function
print(alist) # Print sorted array

# Merge Sort [23]: 

def mergeSort(arr): 
    if len(arr) >1: # If length of array is greater than one then... 
        mid = len(arr)//2 # Determining the "middle" of the array via floor division, i.e. dumps the digits after the decimal
        L = arr[:mid] # Dividing the full array into  the left subarray 
        R = arr[mid:] # Dividing the full array into  the right subarray 
  
        mergeSort(L) # Recursive call to the first half 
        mergeSort(R) # Recursive call to the second half 
  
        i = j = k = 0 # i and j are iterators for the subarrays, k for the original array 
          
        # Copy data to temporary arrays L[] and R[] 
        while i < len(L) and j < len(R): # While (each) index (i) is less than length of L, and index (j) less than length of R
            if L[i] < R[j]:   # If i of L is less than j of R then...
                arr[k] = L[i] # Assign i of L to k or "arr"
                i+=1          # Increment i by one
            else: 
                arr[k] = R[j] # Else assign j of R to k of "arr"
                j+=1          # Increment j by one
            k+=1              # Increment k by one 
          
        # Checking if any element was left 
        while i < len(L): 
            arr[k] = L[i] 
            i+=1
            k+=1
          
        while j < len(R): 
            arr[k] = R[j] 
            j+=1
            k+=1

# Code to print the list 
def printList(arr): 
    for i in range(len(arr)):         
        print(arr[i],end=" ") 
    print() 
   
# Driver code to test the above code
if __name__ == '__main__': 
    arr = [7, 6, 2, 3, 5, 1, 9]  
    print ("Input array is:", end="\n")  
    printList(arr) # Print unsorted array
    mergeSort(arr) # Calling algorithm
    print("Sorted array is: ", end="\n") 
    printList(arr) # Print sorted array

# Counting Sort [24]:

def countingSort(array1, max_val):
    
    m = max_val + 1 # m is the assignment operator for the highest value (9) plus one, i.e. 10
    count = [0] * m  # Creating the count array to contain frequeny count of tallied elements        
    
    for a in array1: # Iterative loop to count occurences of each integer
        count[a] += 1  # Increment the value of each counted/indexed integer and input them into count array           
    i = 0 # Initially set i to zero 
    for a in range(m): # Range m is equal to 10, thus the for loop iterates 10 times            
        for c in range(count[a]): # Using inner loop the iterate over the count array to translate unsorted to sorted values
            array1[i] = a # a and i are iterators used to translate/convert count array into output array
            i += 1 # Increment i by one
    return array1 # Return newly sorted array

print(countingSort([1, 2, 9, 3, 2, 4, 2, 3, 7, 1], 9 ))

# Bubble Sort [25]:

def shortBubbleSort(alist):
    exchanges = True # Sets exchanges initially to True
    passnum = len(alist)-1 # Passnum is the temporary holder of second from last value of array length
    while passnum > 0 and exchanges: # As long as passnum variable is greater than 0 then...
       exchanges = False # Don't swap, and...
       for i in range(passnum): # For i in range (4 in this case)
           if alist[i]>alist[i+1]: # If i is greater than the adjacent element (i+1) then... 
               exchanges = True    # Swap their positions
               temp = alist[i]     # temp is the assigned temporary holder of i  
               alist[i] = alist[i+1] # Swap index [0] for index [1], e.g. in the first iteration of for loop
               alist[i+1] = temp # Swap value that is in temp holder with alist value, and shift up by one index position
       passnum = passnum-1 # passnum is swapped, i.e. shifts from value 4 to 3, e.g. to finish the first iteration of for loop

alist=[19,17,10,12,8] # Unsorted array
shortBubbleSort(alist)
print(alist)

# Heap Sort [26]:

def heapify(arr, n, i): # array, n is size of heap, i is holder for largest element 
    largest = i # Initialize largest as root 
    l = 2 * i + 1     # Left = Multiplies root by 2 and adds one. Thus, if root = 7 then: 7 * 7 + 1 = 15 (l = 15) 
    r = 2 * i + 2     # Right = Multiplies root by 2 and adds two 
  
    # See if left child of root exists and is 
    # greater than root: 
    if l < n and arr[i] < arr[l]: # If l is less than n, and i is less than l then...
        largest = l               # assign l as largest (root)
  
    # See if right child of root exists and is 
    # greater than root: 
    if r < n and arr[largest] < arr[r]: # If r is less than n and largest less than r then... 
        largest = r                     # assign r as largest (root)
  
    # Change root, if needed: 
    if largest != i: # If root is not greater than i then...
        arr[i],arr[largest] = arr[largest],arr[i] # swap root 
  
        # Heapify the root. 
        heapify(arr, n, largest) # Make recursive call to heapify the root again
  
# Main function to sort an array of given size 
def heapSort(arr): 
    n = len(arr) # n equals length of array
  
    # Build a maxheap. 
    for i in range(n, -1, -1): # For i in range 
        heapify(arr, n, i) # Make a recursive call to heapify
  
    # One by one extract elements 
    for i in range(n-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i] # Swap elements
        heapify(arr, i, 0) # Make recursive call to heapify
  
# Driver code to test above 
arr = [8,3,2,7,9,1,4] # Unsorted array
heapSort(arr) # Call HeapSort 
n = len(arr) 
print ("Sorted array is") 
for i in range(n): 
    print ("%d" %arr[i])
#-------------------------------------------------------------------------------------------------------------------------------#
print("Warm-up complete, initializing benchmarking procedure...")
#------------------------------------Complete integrated benchmarking program [18], [19].---------------------------------------#

# Import the necessary modules:
import pandas as pd
import time 
import statistics 
import numpy as np
from numpy.random import seed 
from numpy.random import randint 
import matplotlib.pyplot as plt

# ggplot option:
# plt.style.use('ggplot')

# Random seed option: Random seed will initiate the random integer generator (randint) to always reproduce the same 
# random array for each implemented sorting algorithm. However in this case it was felt best to not use a random seed in order 
# to pass in a new random array for every iteration of each algorithm.

# np.random.seed(1234) 

lengths = list([50,100,200,400,500,750,1000,2500,5000,10000]) # Array of input sizes (n)
times = list([]) # Array for containing execution times 

for i in lengths: # loop to iterate over input size (n) list (lengths)
  
    # Generate random integers 
    array = randint(0, 100, i) # First parameter: Lowest and highest integer range (0 to 99); Second parmeter: Input array size 

    # Algorithm timer:
    start = time.time()     # Start timer
    insertionSort(array)    # Begin algorithm
    end = time.time()       # Stop timer
    times.append(end-start) # Append time, generate time in milliseconds

#-------------------------------------------------Algorithm plot----------------------------------------------------------------#

plt.xlabel('Input Size n') # Label x axis
plt.ylabel('Running Time (Milliseconds)')  # Labe; y axis
plt.plot(lengths, times,  '--bo', label ='Insertion Sort') # Plot line points and markers
plt.grid(True) # Add a reference grid 
plt.legend() 
plt.show() 

#-------------------------------------------Timing of different input instances-------------------------------------------------#

# Timing to discern average runtime per input size (from 10 runs per size (lengths)).
# Here, separate timings are conducted for every input instance in lengths array.
# Averaging methods: np.mean(times[]) -OR- sum(times[])/float(len(times))

for q in range(10): # Execute 10 runs of input instances
# n = 50
    a = randint(0, 100, lengths[0]) # Generate random integers ranging 0-99, initiate input of n = 100 i.e. index [o] in "lengths"
    
    start = time.time() 
    insertionSort(a)
    end = time.time()
    times.append(end-start) # Generate run time in milliseconds
    times[0] = np.mean(round(times[0],4)) # Updating times index to the average run of 10 runs and rounded to 4 decimal places

# n = 100
    b = randint(0, 100, lengths[1])
    
    start = time.time() 
    insertionSort(b) 
    end = time.time() 
    times.append(end-start)
    times[1] = np.mean(round(times[1],4))

# n = 200
    c = randint(0, 100, lengths[2]) 
    
    start = time.time() 
    insertionSort(c) 
    end = time.time() 
    times.append(end-start)
    times[2] = np.mean(round(times[2],4))

# n = 400
    d = randint(0, 100, lengths[3]) 
    
    start = time.time() 
    insertionSort(d) 
    end = time.time() 
    times.append(end-start)
    times[3] = np.mean(round(times[3],4))

# n = 500
    e = randint(0, 100, lengths[4]) 
    
    start = time.time() 
    insertionSort(e) 
    end = time.time() 
    times.append(end-start)
    times[4] = np.mean(round(times[4],4))

# n = 750
    f = randint(0, 100, lengths[5]) 
    
    start = time.time() 
    insertionSort(f) 
    end = time.time()  
    times.append(end-start)
    times[5] = np.mean(round(times[5],4))

# n = 1000
    g = randint(0, 100, lengths[6]) 
    
    start = time.time() 
    insertionSort(g) 
    end = time.time()
    times.append(end-start)
    times[6] = np.mean(round(times[6],4))

# n = 2500
    h = randint(0, 100, lengths[7]) 
    
    start = time.time() 
    insertionSort(h) 
    end = time.time() 
    times.append(end-start)
    times[7] = np.mean(round(times[7],4))

# n = 5000
    i = randint(0, 100, lengths[8]) 
    
    start = time.time() 
    insertionSort(i) 
    end = time.time() 
    times.append(end-start)
    times[8] = np.mean(round(times[8],4))

# n = 10000
    j = randint(0, 100, lengths[9]) 
    
    start = time.time() 
    insertionSort(j) 
    end = time.time()
    times.append(end-start)
    times[9] = np.mean(round(times[9],4))
       
print(len(a), "Elements Sorted by Insertion sort over ten runs at an average of", times[0], "milliseconds per run")
print(len(b), "Elements Sorted by Insertion sort over ten runs at an average of", times[1], "milliseconds per run")
print(len(c), "Elements Sorted by Insertion sort over ten runs at an average of", times[2], "milliseconds per run")
print(len(d), "Elements Sorted by Insertion sort over ten runs at an average of", times[3], "milliseconds per run")
print(len(e), "Elements Sorted by Insertion sort over ten runs at an average of", times[4], "milliseconds per run")
print(len(f), "Elements Sorted by Insertion sort over ten runs at an average of", times[5], "milliseconds per run")
print(len(g), "Elements Sorted by Insertion sort over ten runs at an average of", times[6], "milliseconds per run")
print(len(h), "Elements Sorted by Insertion sort over ten runs at an average of", times[7], "milliseconds per run")
print(len(i), "Elements Sorted by Insertion sort over ten runs at an average of", times[8], "milliseconds per run")
print(len(j), "Elements Sorted by Insertion sort over ten runs at an average of", times[9], "milliseconds per run")

#---------------------------------------------Dataframe of algorithm outputs----------------------------------------------------#

# Intialise data of input sizes and running times.
data = {'Input size:':['Insertion Sort'],'50':times[0],'100':times[1], '200':times[2], '400':times[3], 
        '500':times[4],'750':times[5],'1000':times[6],'2500':times[7],'5000':times[8],'10000':times[9]}

# Create DataFrame
df2 = pd.DataFrame(data)
df2
#---------------------------------------------------Merge sort------------------------------------------------------------------#

lengths = list([50,100,200,400,500,750,1000,2500,5000,10000]) # Input sizes
times1 = list([]) # Array for containing execution times 
for i in lengths: 
  
    # Generate random integers 
    array = randint(0, 100, i) # First parameter: Lowest and highest integer range (0 to 99); Second parmeter: Input array size 

    # Algorithm timer:
    start = time.time()        # Start timer
    mergeSort(array)           # Begin algorithm
    end = time.time()          # Stop timer 
    times1.append(end-start)   # Append time
                               
#-------------------------------------------------Algorithm plot----------------------------------------------------------------#

plt.xlabel('Input Size n') 
plt.ylabel('Running Time (Milliseconds)') 
plt.plot(lengths, times1, '-*', label ='Merge Sort')
plt.grid(True) 
plt.legend() 
plt.show()     

#-------------------------------------------Timing of different input instances-------------------------------------------------#

# Timing to discern average runtime per input size (from 10 runs per size (lengths)).
# Here, separate timings are conducted for every input instance in lengths array.
# Averaging methods: np.mean(times[]) -OR- sum(times[])/float(len(times)).

for z in range(10): # Execute 10 runs of input instance
# n = 50
    a = randint(0, 100, lengths[0]) # Generation of random integers 
    
    start = time.time() 
    mergeSort(a)
    end = time.time()
    times1.append(end-start)
    times1[0] = np.mean(round(times1[0],4)) # Determining average run of 10 runs and rounding to 4 decimal places

# n = 100
    b = randint(0, 100, lengths[1])
    
    start = time.time() 
    mergeSort(b) 
    end = time.time() 
    times1.append(end-start)
    times1[1] = np.mean(round(times1[1],4))

# n = 200
    c = randint(0, 100, lengths[2]) 
    
    start = time.time() 
    mergeSort(c) 
    end = time.time() 
    times1.append(end-start)
    times1[2] = np.mean(round(times1[2],4))

# n = 400
    d = randint(0, 100, lengths[3]) 
    
    start = time.time() 
    mergeSort(d) 
    end = time.time() 
    times1.append(end-start)
    times1[3] = np.mean(round(times1[3],4))

# n = 500
    e = randint(0, 100, lengths[4]) 
    
    start = time.time() 
    mergeSort(e) 
    end = time.time() 
    times1.append(end-start)
    times1[4] = np.mean(round(times1[4],4))

# n = 750
    f = randint(0, 100, lengths[5]) 
    
    start = time.time() 
    mergeSort(f) 
    end = time.time() 
    times1.append(end-start)
    times1[5] = np.mean(round(times1[5],4))

# n = 1000
    g = randint(0, 100, lengths[6]) 
    
    start = time.time() 
    mergeSort(g) 
    end = time.time() 
    times1.append(end-start)
    times1[6] = np.mean(round(times1[6],4))

# n = 2500
    h = randint(0, 100, lengths[7]) 
    
    start = time.time() 
    mergeSort(h) 
    end = time.time() 
    times1.append(end-start)
    times1[7] = np.mean(round(times1[7],4))

# n = 5000
    i = randint(0, 100, lengths[8]) 
    
    start = time.time() 
    mergeSort(i) 
    end = time.time() 
    times1.append(end-start)
    times1[8] = np.mean(round(times1[8],4))

# n = 10000
    j = randint(0, 100, lengths[9]) 
    
    start = time.time() 
    mergeSort(j) 
    end = time.time() 
    times1.append(end-start)
    times1[9] = np.mean(round(times1[9],4))
    
print(len(a), "Elements Sorted by Merge sort over ten runs at an average of", times1[0], "milliseconds per run")
print(len(b), "Elements Sorted by Merge sort over ten runs at an average of", times1[1], "milliseconds per run")
print(len(c), "Elements Sorted by Merge sort over ten runs at an average of", times1[2], "milliseconds per run")
print(len(d), "Elements Sorted by Merge sort over ten runs at an average of", times1[3], "milliseconds per run")
print(len(e), "Elements Sorted by Merge sort over ten runs at an average of", times1[4], "milliseconds per run")
print(len(f), "Elements Sorted by Merge sort over ten runs at an average of", times1[5], "milliseconds per run")
print(len(g), "Elements Sorted by Merge sort over ten runs at an average of", times1[6], "milliseconds per run")
print(len(h), "Elements Sorted by Merge sort over ten runs at an average of", times1[7], "milliseconds per run")
print(len(i), "Elements Sorted by Merge sort over ten runs at an average of", times1[8], "milliseconds per run")
print(len(j), "Elements Sorted by Merge sort over ten runs at an average of", times1[9], "milliseconds per run")


#------------------------------------------Update dataframe---------------------------------------------------------------------#

df2.loc[1] = {'Input size:':'Merge Sort','50':times1[0],'100':times1[1],'200': times1[2],'400':times1[3], 
        '500':times1[4],'750':times1[5],'1000':times1[6],'2500':times1[7],'5000': times1[8],'10000':times1[9]}
df2
#--------------------------------------------Counting sort----------------------------------------------------------------------#

lengths = list([50,100,200,400,500,750,1000,2500,5000,10000]) # Input sizes
times2 = list([]) # Array for containing execution times 
for i in lengths: 
  
    # Generate random integers 
    array = randint(0, 100, i) # First parameter: Lowest and highest integer range (0 to 99); Second parmeter: Input array size 

    # Algorithm timer:
    start = time.time()        # Start timer
    countingSort(array, 99)    # Begin algorithm
    end = time.time()          # Stop timer
    times2.append(end-start)   # Append time
                              
#-------------------------------------------------Algorithm plot----------------------------------------------------------------#

plt.xlabel('Input Size n') 
plt.ylabel('Running Time (Milliseconds)') 
plt.plot(lengths, times2, '-p', label ='Counting Sort')
plt.grid(True) 
plt.legend() 
plt.show()     

#-------------------------------------------Timing of different input instances-------------------------------------------------#

# Timing to discern average runtime per input size (from 10 runs per size (lengths)).
# Here, separate timings are conducted for every input instance in lengths array.
# Averaging methods: np.mean(times[]) -OR- sum(times[])/float(len(times)).

for z in range(10): # Execute 10 runs of input instance
# n = 50
    a = randint(0, 100, lengths[0]) # Generation of random integers 
    
    start = time.time() 
    countingSort(a, 99)
    end = time.time()
    times2.append(end-start)
    times2[0] = np.mean(round(times2[0],4)) # Determining average run of 10 runs and rounding to 4 decimal places

# n = 100
    b = randint(0, 100, lengths[1])
    
    start = time.time() 
    countingSort(b, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[1] = np.mean(round(times2[1],4))

# n = 200
    c = randint(0, 100, lengths[2]) 
    
    start = time.time() 
    countingSort(c, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[2] = np.mean(round(times2[2],4))

# n = 400
    d = randint(0, 100, lengths[3]) 
    
    start = time.time() 
    countingSort(d, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[3] = np.mean(round(times2[3],4))

# n = 500
    e = randint(0, 100, lengths[4]) 
    
    start = time.time() 
    countingSort(e, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[4] = np.mean(round(times2[4],4))

# n = 750
    f = randint(0, 100, lengths[5]) 
    
    start = time.time() 
    countingSort(f, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[5] = np.mean(round(times2[5],4))

# n = 1000
    g = randint(0, 100, lengths[6]) 
    
    start = time.time() 
    countingSort(g, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[6] = np.mean(round(times2[6],4))

# n = 2500
    h = randint(0, 100, lengths[7]) 
    
    start = time.time() 
    countingSort(h, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[7] = np.mean(round(times2[7],4))

# n = 5000
    i = randint(0, 100, lengths[8]) 
    
    start = time.time() 
    countingSort(i, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[8] = np.mean(round(times2[8],4))

# n = 10000
    j = randint(0, 100, lengths[9]) 
    
    start = time.time() 
    countingSort(j, 99) 
    end = time.time() 
    times2.append(end-start)
    times2[9] = np.mean(round(times2[9],4))
        
print(len(a), "Elements Sorted by Counting sort over ten runs at an average of", times2[0], "milliseconds per run")
print(len(b), "Elements Sorted by Counting sort over ten runs at an average of", times2[1], "milliseconds per run")
print(len(c), "Elements Sorted by Counting sort over ten runs at an average of", times2[2], "milliseconds per run")
print(len(d), "Elements Sorted by Counting sort over ten runs at an average of", times2[3], "milliseconds per run")
print(len(e), "Elements Sorted by Counting sort over ten runs at an average of", times2[4], "milliseconds per run")
print(len(f), "Elements Sorted by Counting sort over ten runs at an average of", times2[5], "milliseconds per run")
print(len(g), "Elements Sorted by Counting sort over ten runs at an average of", times2[6], "milliseconds per run")
print(len(h), "Elements Sorted by Counting sort over ten runs at an average of", times2[7], "milliseconds per run")
print(len(i), "Elements Sorted by Counting sort over ten runs at an average of", times2[8], "milliseconds per run")
print(len(j), "Elements Sorted by Counting sort over ten runs at an average of", times2[9], "milliseconds per run")


#-----------------------------------------------Update dataframe----------------------------------------------------------------#

df2.loc[2] = {'Input size:':'Counting Sort','50':times2[0],'100':times2[1],'200': times2[2],'400':times2[3], 
        '500':times2[4],'750':times2[5],'1000':times2[6],'2500':times2[7],'5000': times2[8],'10000':times2[9]}
df2
#-------------------------------------------------Bubble sort------------------------------------------------------------------#

lengths = list([50,100,200,400,500,750,1000,2500,5000,10000]) # Input sizes
times3 = list([]) # Array for containing execution times 
for i in lengths: 
  
    # Generate random integers 
    array = randint(0, 100, i) # First parameter: Lowest and highest integer range (0 to 99); Second parmeter: Input array size 

    # Algorithm timer:
    start = time.time()       # Start timer
    shortBubbleSort(array)    # Begin algorithm
    end = time.time()         # Stop timer 
    times3.append(end-start)  # Append time
                            
#-------------------------------------------------Algorithm plot----------------------------------------------------------------#

plt.xlabel('Input Size n') 
plt.ylabel('Running Time (Milliseconds)') 
plt.plot(lengths, times3, '-*', label ='Bubble sort')
plt.grid(True) 
plt.legend() 
plt.show()     

#-------------------------------------------Timing of different input instances-------------------------------------------------#

# Timing to discern average runtime per input size (from 10 runs per size (lengths)).
# Here, separate timings are conducted for every input instance in lengths array.
# Averaging methods: np.mean(times[]) -OR- sum(times[])/float(len(times)).

for z in range(10): # Execute 10 runs of input instance
# n = 50
    a = randint(0, 100, lengths[0]) # Generation of random integers 
    
    start = time.time() 
    shortBubbleSort(a)
    end = time.time()
    times3.append(end-start)
    times3[0] = np.mean(round(times3[0],4)) # Determining average run of 10 runs and rounding to 4 decimal places

# n = 100
    b = randint(0, 100, lengths[1])
    
    start = time.time() 
    shortBubbleSort(b) 
    end = time.time() 
    times3.append(end-start)
    times3[1] = np.mean(round(times3[1],4))

# n = 200
    c = randint(0, 100, lengths[2]) 
    
    start = time.time() 
    shortBubbleSort(c) 
    end = time.time() 
    times3.append(end-start)
    times3[2] = np.mean(round(times3[2],4))

# n = 400
    d = randint(0, 100, lengths[3]) 
    
    start = time.time() 
    shortBubbleSort(d) 
    end = time.time() 
    times3.append(end-start)
    times3[3] = np.mean(round(times3[3],4))

# n = 500
    e = randint(0, 100, lengths[4]) 
    
    start = time.time() 
    shortBubbleSort(e) 
    end = time.time() 
    times3.append(end-start)
    times3[4] = np.mean(round(times3[4],4))

# n = 750
    f = randint(0, 100, lengths[5]) 
    
    start = time.time() 
    shortBubbleSort(f) 
    end = time.time()
    lengths.append(len(f))
    times3.append(end-start)
    times3[5] = np.mean(round(times3[5],4))

# n = 1000
    g = randint(0, 100, lengths[6]) 
    
    start = time.time() 
    shortBubbleSort(g) 
    end = time.time() 
    times3.append(end-start)
    times3[6] = np.mean(round(times3[6],4))

# n = 2500
    h = randint(0, 100, lengths[7]) 
    
    start = time.time() 
    shortBubbleSort(h) 
    end = time.time() 
    times3.append(end-start)
    times3[7] = np.mean(round(times3[7],4))

# n = 5000
    i = randint(0, 100, lengths[8]) 
    
    start = time.time() 
    shortBubbleSort(i) 
    end = time.time() 
    lengths.append(len(i))
    times3.append(end-start)
    times3[8] = np.mean(round(times3[8],4))

# n = 10000
    j = randint(0, 100, lengths[9]) 
    
    start = time.time() 
    shortBubbleSort(j) 
    end = time.time() 
    times3.append(end-start)
    times3[9] = np.mean(round(times3[9],4))
       
print(len(a), "Elements Sorted by Bubble sort over ten runs at an average of", times3[0], "milliseconds per run")
print(len(b), "Elements Sorted by Bubble sort over ten runs at an average of", times3[1], "milliseconds per run")
print(len(c), "Elements Sorted by Bubble sort over ten runs at an average of", times3[2], "milliseconds per run")
print(len(d), "Elements Sorted by Bubble sort over ten runs at an average of", times3[3], "milliseconds per run")
print(len(e), "Elements Sorted by Bubble sort over ten runs at an average of", times3[4], "milliseconds per run")
print(len(f), "Elements Sorted by Bubble sort over ten runs at an average of", times3[5], "milliseconds per run")
print(len(g), "Elements Sorted by Bubble sort over ten runs at an average of", times3[6], "milliseconds per run")
print(len(h), "Elements Sorted by Bubble sort over ten runs at an average of", times3[7], "milliseconds per run")
print(len(i), "Elements Sorted by Bubble sort over ten runs at an average of", times3[8], "milliseconds per run")
print(len(j), "Elements Sorted by Bubble sort over ten runs at an average of", times3[9], "milliseconds per run")


#-----------------------------------------Update dataframe---------------------------------------------------------------------#

df2.loc[3] = {'Input size:':'Bubble Sort','50':times3[0],'100':times3[1],'200': times3[2],'400':times3[3], 
        '500':times3[4],'750':times3[5],'1000':times3[6],'2500':times3[7],'5000': times3[8],'10000':times3[9]}
df2
#-------------------------------------------------Heap sort--------------------------------------------------------------------#

lengths = list([50,100,200,400,500,750,1000,2500,5000,10000]) # Input sizes
times4 = list([]) # Array for containing execution ti4mes 
for i in lengths: 
  
    # Generate random integers 
    array = randint(0, 100, i) # First parameter: Lowest and highest integer range (0 to 99); Second parmeter: Input array size 

    # Algorithm timer:
    start = time.time()      # Start timer
    heapSort(array)          # Begin algorithm
    end = time.time()        # Stop timer
    times4.append(end-start) # Append time
                
#-------------------------------------------------Algorithm plot----------------------------------------------------------------#

plt.xlabel('Input Size n') 
plt.ylabel('Running Time (Milliseconds)') 
plt.plot(lengths, times4, '--s', label ='Heap Sort')
plt.grid(True) 
plt.legend() 
plt.show()     

#-------------------------------------------Timing of different input instances-------------------------------------------------#

# Timing to discern average runtime per input size (from 10 runs per size (lengths)).
# Here, separate timings are conducted for every input instance in lengths array.
# Averaging methods: np.mean(times[]) -OR- sum(times[])/float(len(times)).

for z in range(10): # Execute 10 runs of input instance
# n = 50
    a = randint(0, 100, lengths[0]) # Generation of random integers 
    
    start = time.time() 
    heapSort(a)
    end = time.time()
    times4.append(end-start)
    times4[0] = np.mean(round(times4[0],4)) # Determining average run of 10 runs and rounding to 4 decimal places

# n = 100
    b = randint(0, 100, lengths[1])
    
    start = time.time() 
    heapSort(b) 
    end = time.time() 
    times4.append(end-start)
    times4[1] = np.mean(round(times4[1],4))

# n = 200
    c = randint(0, 100, lengths[2]) 
    
    start = time.time() 
    heapSort(c) 
    end = time.time() 
    times4.append(end-start)
    times4[2] = np.mean(round(times4[2],4))

# n = 400
    d = randint(0, 100, lengths[3]) 
    
    start = time.time() 
    heapSort(d) 
    end = time.time() 
    times4.append(end-start)
    times4[3] = np.mean(round(times4[3],4))

# n = 500
    e = randint(0, 100, lengths[4]) 
    
    start = time.time() 
    heapSort(e) 
    end = time.time()
    times4.append(end-start)
    times4[4] = np.mean(round(times4[4],4))

# n = 750
    f = randint(0, 100, lengths[5]) 
    
    start = time.time() 
    heapSort(f) 
    end = time.time() 
    times4.append(end-start)
    times4[5] = np.mean(round(times4[5],4))

# n = 1000
    g = randint(0, 100, lengths[6]) 
    
    start = time.time() 
    heapSort(g) 
    end = time.time()
    times4.append(end-start)
    times4[6] = np.mean(round(times4[6],4))

# n = 2500
    h = randint(0, 100, lengths[7]) 
    
    start = time.time() 
    heapSort(h) 
    end = time.time() 
    times4.append(end-start)
    times4[7] = np.mean(round(times4[7],4))

# n = 5000
    i = randint(0, 100, lengths[8]) 
    
    start = time.time() 
    heapSort(i) 
    end = time.time() 
    times4.append(end-start)
    times4[8] = np.mean(round(times4[8],4))

# n = 10000
    j = randint(0, 100, lengths[9]) 
    
    start = time.time() 
    heapSort(j) 
    end = time.time() 
    times4.append(end-start)
    times4[9] = np.mean(round(times4[9],4))
    
print(len(a), "Elements Sorted by Heap sort over ten runs at an average of", times4[0], "milliseconds per run")
print(len(b), "Elements Sorted by Heap sort over ten runs at an average of", times4[1], "milliseconds per run")
print(len(c), "Elements Sorted by Heap sort over ten runs at an average of", times4[2], "milliseconds per run")
print(len(d), "Elements Sorted by Heap sort over ten runs at an average of", times4[3], "milliseconds per run")
print(len(e), "Elements Sorted by Heap sort over ten runs at an average of", times4[4], "milliseconds per run")
print(len(f), "Elements Sorted by Heap sort over ten runs at an average of", times4[5], "milliseconds per run")
print(len(g), "Elements Sorted by Heap sort over ten runs at an average of", times4[6], "milliseconds per run")
print(len(h), "Elements Sorted by Heap sort over ten runs at an average of", times4[7], "milliseconds per run")
print(len(i), "Elements Sorted by Heap sort over ten runs at an average of", times4[8], "milliseconds per run")
print(len(j), "Elements Sorted by Heap sort over ten runs at an average of", times4[9], "milliseconds per run")


#------------------------------------------------Update dataframe---------------------------------------------------------------#

df2.loc[4] = {'Input size:':'Heap Sort','50':times4[0],'100':times4[1],'200': times4[2],'400':times4[3], 
        '500':times4[4],'750':times4[5],'1000':times4[6],'2500':times4[7],'5000': times4[8],'10000':times4[9]}
df2
#--------------------------------------Multiple line plots for all algorithms---------------------------------------------------#

lengths = [50,100,200,400,500,750,1000,2500,5000,10000]
times = [] 
for i in lengths: 
  
    # generate some integers 
    a = randint(0, 100, i) 
    
    # Timer:
    start = time.time()  
    insertionSort(a) 
    end = time.time() 
    times.append(end-start)
    
times1 = [] 
for i in lengths: 
  
    a = randint(0, 100, i) 
    
    # Timer:
    start = time.time()  
    mergeSort(a) 
    end = time.time()
    times1.append(end-start)
    
times2 = [] 
for i in lengths: 
  
    a = randint(0, 100, i) 

    # Timer:
    start = time.time() 
    countingSort(a, 99) 
    end = time.time()
    times2.append(end-start)
    
times3 = [] 
for i in lengths: 
  
    a = randint(0, 100, i) 
    
    # Timer:
    start = time.time() 
    shortBubbleSort(a) 
    end = time.time()
    times3.append(end-start)
    
times4 = [] 
for i in lengths: 
  
    a = randint(0, 100, i) 
    
    # Timer:
    start = time.time() 
    heapSort(a) 
    end = time.time()
    times4.append(end-start)
    
  
plt.xlabel('Input Size n') 
plt.ylabel('Running Time (Milliseconds)') 
plt.plot(lengths, times4, '--s', label ='Heap Sort')
plt.plot(lengths, times3, '-v', label ='Bubble Sort')
plt.plot(lengths, times2, '-p', label ='Counting Sort')
plt.plot(lengths, times1, '-*', label ='Merge Sort') 
plt.plot(lengths, times, '--bo', label ='Insertion Sort') 
plt.grid(True) 
plt.legend() 
plt.show() 

#-------------------------------- Comparison of project output with typical growth complexities---------------------------------#

# Comparison time complexity chart [19].

# Import libraries:
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
# %precision 4
plt.style.use('ggplot')

print("Valid growth order chart representing time complexities, coded for comparison purposes:")

# Growth orders:

# Generally, 'n' is the number of elements currently in the container. 
# 'k' is either the value of a parameter or the number of elements in the parameter. 

# Quadratic
def f1(n, k):
    return k*n*n
# Linearithmic
def f2(n, k):
    return k*n*np.log(n)
# linear 
def f3(n,k):
    return k*n
# Logarithmic 
def f4(n, k):
    return k*np.log(n)
# Cubic
def f5(n, k):
    return k*n*n*n
# Exponential
def f6(n, k):
    return k*2*n


n = np.arange(0, 10000) # Input size arrangement/threshold

plt.plot(n, f1(n, 1.5), c='blue')
plt.plot(n, f2(n, 700), c='red')
plt.plot(n, f3(n, 2000), c='green')
plt.plot(n, f4(n, 2000), c='yellow')
plt.plot(n, f5(n, .0002), c='purple')
plt.plot(n, f6(n, 10000), c='black')
plt.xlabel('Input size (n)', fontsize=10)
plt.ylabel('Number of operations / Runtime', fontsize=10)
plt.legend(['Quadratic: $\mathcal{O}(n^2)$', 'Linearithmic: $\mathcal{O}(n \log n)$', 'Linear: $\mathcal{O}(n)$', 
            'Logarithmic:$\mathcal{O}(log n)$', 'Cubic: $\mathcal{O}(n^3)$', 'Exponential: $\mathcal{O}(2n)$'], 
           loc='best', fontsize=9);
plt.show() 

#-----------------------------------------------Output final dataframe----------------------------------------------------------#
# Dataframe:
print(df2)
#--------------------------------------------------Program finished-------------------------------------------------------------#
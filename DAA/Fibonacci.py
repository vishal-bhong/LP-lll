#with recursion
def recur_fibo(n):  
   if n <= 1:  
       return n  
   else:  
       return(recur_fibo(n-1) + recur_fibo(n-2))  
# take input from the user  
nterms = int(input("How many terms? "))  
# check if the number of terms is valid  
if nterms <= 0:  
   print("Plese enter a positive integer")  
else:  
   print("Fibonacci sequence:")  
   for i in range(nterms):  
       print(recur_fibo(i)) 

#without recursion

n1=0
n2=1
n=10
for i in range(1,n):
    n3=n1+n2
    n1=n2
    n2=n3
    print(n3,end=" ")

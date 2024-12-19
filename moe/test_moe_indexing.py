import torch 

input_dim = 10 
output_dim = 9
batch_size = 8 
num_experts = 5 
K = 6

## input data 
D = torch.randn(batch_size, input_dim) 
print('D')
print(D) 

## output of router 
l = torch.randn(batch_size, num_experts) 
print('l')
print(l)

## perform topk on router results 
a, ib = torch.topk(l, K, dim=0)
print('a') 
print(a) 
print('ib') 
print(ib) 

## index select on input data [for first expert] 
selected = torch.index_select(D, 0, ib[:, 0]) 
print('selected') 
print(selected) 

# forward the data
## n/a 

output = torch.zeros((batch_size, input_dim)) 
# fill in the output 
output[ib[:, 0], :] += selected 

print('output')
print(output)



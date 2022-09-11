import mysearchsys as mss

# procedure for assembling a collection from files in the `/text` directory
# mss.make_collection()

# collection search function
# returns a tuple:
#   (list of docs when tf = count, list of docs when tf = log(1+count))
# mss.search('На складе чугуноплавильного завода скопился годовой запас продукции')

# example
#   if first index is [0] then tf = count
#   if - - - - - - -  [1] then tf = log(1+count)
search_list = mss.search('Внешность короля Генриха III')[0][:3]

print(search_list)

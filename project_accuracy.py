def accuracy(prediction, label, batch_size, window_size = 1):
    #check if window_size is an integer
    #convert each 3D tensor into a 3D numpy array
    pred_np = prediction.numpy()
    label_np = label.numpy() 
    #label_np = np.where(label_np > 0.5, 1, 0)
    #check that the dimensions of the prediction and the label are the same
    if pred_np.shape != label_np.shape:
        print("Error- prediction and label sizes don't match!")
        return
    #check that window_size is a positive odd integer 
    if window_size <= 0:
        print("Error- window_size is not positive!")
        return
    elif isinstance(window_size, int) != True:
        print("Error- window_size is not an integer!")
        return
    elif isinstance(window_size, int) == True:
        if window_size % 2 == 0:
            print("Error- window_size is not odd!")
            return
    #check if the values in each column of the prediction are the same for the label within the window_size
    #initialize count
    #find the number of frames of each file
    col = pred_np.shape[2]
    count = 0
    #count = np.zeros((col, batch_size))
    for i in range(col):
        #print('i:', i)
    #case where window_size is out of bounds (beginning case)
        if i >=0 and i < int(window_size/2):
            lower_lim = 0
            upper_lim = i + int(window_size/2)
        #case where window_size is out of bounds (end case)
        elif i >= col - int(window_size/2) and i <= col:
            lower_lim = i - int(window_size/2)
            upper_lim = col - 1
        else:
            lower_lim = i - int(window_size/2)
            upper_lim = i + int(window_size/2)
        #print(lower_lim, upper_lim)
        #numpy array transpose
        #print('non-transpose')
        #print(pred_np)
        #print(label_np)
        p = pred_np.transpose(0, 2, 1)
        l_w = label_np.transpose(0, 2, 1)
        #print('p')
        #print(p)
        #print('l_w')
        #print(l_w)
        for j in range(batch_size):
          #print('j:', j)
          window_arr = l_w[j][lower_lim:upper_lim + 1]
          #print('window')
          #print(window_arr.tolist())
          #print('pj')
          #print(p[j][i].tolist())
          if (p[j][i].tolist() in window_arr.tolist()) == True: 
          #if (p[j][:, i:i+1].tolist() in window_arr.tolist()) == True: 
            count += 1
    avg_acc = count/(col*batch_size)
    print("Batch Accuracy:", avg_acc)
    return avg_acc
tst = data_gen(r"D:\RFtest",5,60)

    [X_sig, y_sig] = loadData(tst)

    X_sig = np.array(X_sig)
    y_sig = np.array(y_sig)

    X_sig = sobelFilter(X_sig)

    X_nu = np.zeros(shape=(X_sig.shape[0], (X_sig.shape[1])**2))
    print(X_nu.shape)
    for i in range(X_sig.shape[0]):
        X_nu[i,:] = X_sig[i,:,:].flatten()

    print("X_sig", X_nu.shape)
    print(X_nu.shape[0])
    print(X_nu.shape[1])
    print(y_sig.shape)

    recall = cross_val_score(rf, X_nu, y_sig, cv=5, scoring='recall')
    precision = cross_val_score(rf, X_nu, y_sig, cv=5, scoring='precision')
    accuracy = cross_val_score(rf, X_nu, y_sig, cv=5, scoring='accuracy')
    f1_score = cross_val_score(rf, X_nu, y_sig, cv=5, scoring='f1_macro')

    print("Precision, acc", precision, accuracy)
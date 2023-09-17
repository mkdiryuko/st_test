import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

def create_Xy(weights, bias, test_size):
    start = 0
    end = 1
    step = 0.01
    
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weights * X + bias
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42, shuffle=False)
    
    return X_train, X_test, y_train, y_test

def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None):
    
    """
    Plots training data, test data and compares predictions.
    """
    
    fig, ax = plt.subplots()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1,1,1)
    
    # Plot traing data in blue
    ax.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    ax.scatter(test_data, test_labels,c="g", s=4, label="Testing data")
    
    ax.set_title("Linear function")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Are there predictions?
    if predictions is not None:
        # Plot the predictions if they exist
        ax.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    ax.legend(prop={"size": 14})
    
    return st.pyplot(fig)

def Loss_curves(train_loss, test_loss, epochs):
    fig, ax = plt.subplots()
    
    ax.plot(epochs, np.array(torch.tensor(train_loss).numpy()), label='Train loss')
    ax.plot(epochs, np.array(torch.tensor(test_loss).numpy()), label='Test loss')
    ax.set_title("Training and Test loss curves")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    
    return st.pyplot(fig)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
def main():
    st.set_page_config(layout="centered")
    # タイトルの表示
    st.title("LinearRegressionModel")
    # 制作者の表示
    st.text("Created by Yuto Kobayashi")
    # アプリの説明の表示
    st.header("概要")

    st.text(" 以下の線形関数の学習を行います")

    st.latex(r'y = weight * x + bias')
    st.text("""
    <学習対象>
    傾き : weight 、切片 : bias 
    
    <用語の説明>
    train : 学習データ（x, y）
    test : 教師データ (x, y)
    predictions : 予測値
    """)
    
    st.header("パラメーターの設定")
    
    weight = st.number_input("##### weight", value = 0.70)
    bias = st.number_input("##### bias", value = 0.30)
    
    train = st.number_input("##### train [%]", min_value = 1, value = 80)
    test = st.number_input("##### test [%] ", min_value = 100-train, max_value = 100-train)
    
    X_train, X_test, y_train, y_test = create_Xy(weight, bias, test*0.01)
    
    model = LinearRegressionModel()
    with torch.inference_mode():
        y_preds = model(X_test)
        
    # train, test の描画
    plot_predictions(X_train, y_train, X_test, y_test, y_preds)
    st.write("##### Predictions の初期値：", model.state_dict())
    
    st.markdown("### 学習")
    epochs = st.number_input("##### epochs", 1, value = 100)
    st.markdown("""
    #### Loss function (損失関数)
    - MAE (平均絶対誤差) (https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
    #### optimizer (最適化手法)
    - SGD (確率的勾配降下法) (https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    """)
    
    torch.manual_seed(42)
    model = LinearRegressionModel()

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    
    # Track different values
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    
    if st.button("学習を始める"):
        for epoch in range(epochs):
            ### Training
            model.train()
            
            # 1. Forward pass
            y_pred = model(X_train)
            # 2. Caluculate the loss
            loss = loss_fn(y_pred, y_train)
            # 3. Optimizer zero
            optimizer.zero_grad()
            # 4. Perform backpropargation
            loss.backward()
            # 5. optimizer.step
            optimizer.step()
            
            ### Testing
            model.eval()
            with torch.inference_mode():
                test_pred = model(X_test)
                
                test_loss = loss_fn(test_pred, y_test)
                
                epoch_count.append(epoch)
                train_loss_values.append(loss)
                test_loss_values.append(test_loss)
                
        st.header("学習結果")
        model.eval()
        plot_predictions(X_train, y_train, X_test, y_test, test_pred)
        Loss_curves(train_loss_values, test_loss_values, epoch_count)
        
if __name__ == "__main__":
    main()
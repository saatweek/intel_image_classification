# from comparisons import agent
import time

from data_prep import train_dir, wnn_preprocessing, validation_dir
import numpy as np
import ao_arch as ar
import ao_core as ao
import pandas as pd


if __name__=="__main__":
    # Training Preprocessing
    training_input, training_output = wnn_preprocessing(train_dir, "cnn_model.pth")

    # Validation Preprocessing
    test_input, test_output = wnn_preprocessing(validation_dir, "cnn_model.pth")

    # weightless neural network architecture
    arch_i = [8 for i in range(256)]
    arch_z = [6]
    arch_c = []
    connections = ["full_conn", "forward_full_conn", "forward_forward_conn", "rand_conn"]
    iterations = [i for i in range(3, 6)]
    connector_parameters = [256*6, 256*4, 256, 6]

    #
    # connector_function = "full_conn"
    # description = "On top of CNN"
    # arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)

    # agent = ao.Agent(arch, notes="cnn_x_wnn", save_meta=False, _steps=50000)
    # agent.full_conn_compress = True
    # agent.next_state_batch(training_input, training_output, DD=True, unsequenced=True)
    #
    # agent.pickle()
    # test_output_wnn = []
    # for idx in range(test_input.shape[0]):
    #     agent.reset_state()
    #     for s in range(3):
    #         res = agent.next_state(test_input[idx], DD=False)
    #     print(f"{idx} of {test_input.shape[0]} done")
    #     test_output_wnn.append(res)
    # test_output_wnn = np.asarray(test_output_wnn)
    # print(f"test_output_wnn.shape : {test_output_wnn.shape}")
    # print(f"test_output_wnn [:2] : {test_output_wnn[:2]}")
    #
    # print(f"Accuracy of WNN model is : {(test_output * test_output_wnn).sum() / test_output.shape[0]}")


    df_conn = []
    df_iter = []
    df_full_conn_compress = []
    df_reset = []
    df_time = []
    df_accuracy = []
    for conn in connections:
        for iter in iterations:
            for reset in [True, False]:
                start_time = time.time()
                arch = ar.Arch(arch_i, arch_z, arch_c, conn, connector_parameters)
                agent = ao.Agent(arch, notes=conn+"_iter"+str(iter)+"_reset_"+str(reset), save_meta=False, _steps=50000)
                df_conn.append(conn)
                df_iter.append(iter)
                df_reset.append("True" if reset else "False")
                try:
                    agent.full_conn_compress = True
                    df_full_conn_compress.append("True")
                except:
                    df_full_conn_compress.append("False")
                agent.next_state_batch(training_input, training_output, unsequenced=True)
                agent.pickle()
                test_output_wnn =[]
                print(f"Processing {conn} with reset={reset}...")
                for idx in range(100):
                    print(f"Processing {idx+1} of 100")
                    if reset:
                        agent.reset_state()
                    for s in range(iter):
                        res = agent.next_state(test_input[idx])
                    test_output_wnn.append(res)
                test_output_wnn = np.asarray(test_output_wnn)
                accuracy = (test_output[:100] * test_output_wnn).sum() / 100
                df_accuracy.append(accuracy)
                df_time.append(time.time()-start_time)
                print(f"Accuracy of {conn} connection with {iter} iteration and reset={reset} is {accuracy}")

    df = pd.DataFrame({"Connection":df_conn, "Iterations":df_iter, "Full_Connection_Compress":df_full_conn_compress, "reset_state":df_reset, "time_taken":df_time, "accuracy":df_accuracy})
    df.to_csv("benchmark.csv", index=False)





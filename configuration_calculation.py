def calculate_amf_memory(ues):
    amf_global_mem = ues * 0.000352 + 0.0377
    amf_packet_mem = (0.000939 + 0.0199) * ues
    amf_session_mem = ((0.000807 + 0.00231) * ues) + 0.00728
    amf_transaction_mem = ues * 0.000168 + 0.000592
    return amf_global_mem, amf_packet_mem, amf_session_mem, amf_transaction_mem

def calculate_smf_memory(ues):
    smf_global_mem = 0.000192 * ues + 0.0657
    smf_packet_mem = ((0.000222 + 0.0275) * ues) + 0.000372
    smf_session_mem = ((0.0333 + 0.0477) * ues + 0.08)
    smf_transaction_mem = 0.00131
    return smf_global_mem, smf_packet_mem, smf_session_mem, smf_transaction_mem

def calculate_ausf_memory(ues):
    ausf_global_mem = 0.0422 + 0.000192 * ues
    ausf_packet_mem = 0
    ausf_session_mem = 0.00139 if ues > 0 else 0
    ausf_transaction_mem = 0.000112 if ues > 0 else 0
    return ausf_global_mem, ausf_packet_mem, ausf_session_mem, ausf_transaction_mem

def calculate_upf1_memory(ues):
    upf1_deployment_mem = 0.058047
    upf1_packet_mem = 0.00064 * ues
    upf1_session_mem = (0.0216 + 0.02468 * ues) * 1.7793
    upf1_transaction_mem = 0.000112
    return upf1_deployment_mem, upf1_packet_mem, upf1_session_mem, upf1_transaction_mem

def calculate_udm_memory(ues):
    udm_deployment_mem = 0.0360 + 0.000128 * ues
    udm_packet_mem = 0
    udm_session_mem = 0
    udm_transaction_mem = 0.000112
    return udm_deployment_mem, udm_packet_mem, udm_session_mem, udm_transaction_mem

def main():
    ues = int(input("How many UEs do you want to accommodate?: "))
    
    print("Calculating memory requirements for AMF:")
    amf_global_mem, amf_packet_mem, amf_session_mem, amf_transaction_mem = calculate_amf_memory(ues)
    print("Total memory needed for AMF global memory pool:", "{:.5f}".format(amf_global_mem), "MB (", "{:.5f}".format(amf_global_mem * 1024), "KB)")
    print("Total memory needed for AMF packet memory pool:", "{:.5f}".format(amf_packet_mem), "MB (", "{:.5f}".format(amf_packet_mem * 1024), "KB)", "(chunk size 2048)")
    print("Total memory needed for AMF session memory pool:", "{:.5f}".format(amf_session_mem), "MB (", "{:.5f}".format(amf_session_mem * 1024), "KB)")
    print("Total memory needed for AMF transaction memory pool:", "{:.5f}".format(amf_transaction_mem), "MB (", "{:.5f}".format(amf_transaction_mem * 1024), "KB)")

    print("\nCalculating memory requirements for SMF:")
    smf_global_mem, smf_packet_mem, smf_session_mem, smf_transaction_mem = calculate_smf_memory(ues)
    print("Total memory needed for SMF global memory pool:", "{:.5f}".format(smf_global_mem), "MB (", "{:.5f}".format(smf_global_mem * 1024), "KB)")
    print("Total memory needed for SMF packet memory pool:", "{:.5f}".format(smf_packet_mem), "MB (", "{:.5f}".format(smf_packet_mem * 1024), "KB)", "(chunk size 4096)")
    print("Total memory needed for SMF session memory pool:", "{:.5f}".format(smf_session_mem), "MB (", "{:.5f}".format(smf_session_mem * 1024), "KB)")
    print("Total memory needed for SMF transaction memory pool during deploument:", "{:.5f}".format(smf_transaction_mem), "MB (", "{:.5f}".format(smf_transaction_mem * 1024), "KB)")

    print("\nCalculating memory requirements for AUSF:")
    ausf_global_mem, ausf_packet_mem, ausf_session_mem, ausf_transaction_mem = calculate_ausf_memory(ues)
    print("Total memory needed for AUSF global memory pool during deployment:", "{:.5f}".format(ausf_global_mem), "MB (", "{:.5f}".format(ausf_global_mem * 1024), "KB)")
    print("Total memory needed for AUSF packet memory pool:", "{:.5f}".format(ausf_packet_mem), "MB (", "{:.5f}".format(ausf_packet_mem * 1024), "KB)")
    print("Total memory needed for AUSF session memory pool during deployment:", "{:.5f}".format(ausf_session_mem), "MB (", "{:.5f}".format(ausf_session_mem * 1024), "KB)")
    print("Total memory needed for AUSF transaction memory pool during deployment:", "{:.5f}".format(ausf_transaction_mem), "MB (", "{:.5f}".format(ausf_transaction_mem * 1024), "KB)")

    print("\nCalculating memory requirements for UPF1:")
    upf1_deployment_mem, upf1_packet_mem, upf1_session_mem, upf1_transaction_mem = calculate_upf1_memory(ues)
    print("Total memory needed for UPF1 global memory pool during deployment:", "{:.5f}".format(upf1_deployment_mem), "MB (", "{:.5f}".format(upf1_deployment_mem * 1024), "KB)")
    print("Total memory needed for UPF1 packet memory pool:", "{:.5f}".format(upf1_packet_mem), "MB (", "{:.5f}".format(upf1_packet_mem * 1024), "KB)")
    print("Total memory needed for UPF1 session memory pool:", "{:.5f}".format(upf1_session_mem), "MB (", "{:.5f}".format(upf1_session_mem * 1024), "KB)")
    print("Total memory needed for UPF1 transaction memory pool:", "{:.5f}".format(upf1_transaction_mem), "MB (", "{:.5f}".format(upf1_transaction_mem * 1024), "KB)")

    print("\nCalculating memory requirements for UDM:")
    udm_deployment_mem, udm_packet_mem, udm_session_mem, udm_transaction_mem = calculate_udm_memory(ues)
    print("Total memory needed for UDM global memory pool:", "{:.5f}".format(udm_deployment_mem), "MB (", "{:.5f}".format(udm_deployment_mem * 1024), "KB)")
    print("Total memory needed for UDM packet memory pool during deployment:", "{:.5f}".format(udm_packet_mem), "MB (", "{:.5f}".format(udm_packet_mem * 1024), "KB)")
    print("Total memory needed for UDM session memory pool during deployment:", "{:.5f}".format(udm_session_mem), "MB (", "{:.5f}".format(udm_session_mem * 1024), "KB)")
    print("Total memory needed for UDM transaction memory pool during deployment:", "{:.5f}".format(udm_transaction_mem), "MB (", "{:.5f}".format(udm_transaction_mem * 1024), "KB)")

if __name__ == "__main__":
    main()

import torch,os, copy
from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
from alora_intrinsics.alora.tokenize_alora import tokenize_alora
int_names = ["rag"]
INVOCATION_PROMPT = "<|start_of_role|>assistant<|end_of_role|>"






token = os.getenv("HF_MISTRAL_TOKEN")
BASE_NAME = "ibm-granite/granite-3.1-8b-instruct"#'/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a'#"ibm-granite/granite-3.0-8b-instruct"
LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/RAG_alora_sz32_long6"#+ int_name 


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")




model_alora = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0], response_token_ids = None) #response_token_ids)
for intname in int_names[1:]:
    model_alora.load_adapter(LORA_NAME + intname, adapter_name = intname)
model_alora.set_adapter("rag")










#system_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior." 


#################################3
##################################
#question = "What is IBM?"#"How can I steal a car?" "
#################################
#################################

rag_chat = {"messages":[{"role":"system","content":"You are an AI assistant who is provided with a conversation between a user and an agent. Your task is to generate agent's response to the last user utterance. You are also provided with multiple documents. These documents may or may not contain the answer. You may or may not use the information in them to generate your response."},{"role":"user","content":"What is the role of the for loop in IBM Storage Virtualize CLI and how is it used?"},{"role":"assistant","content":"The for loop in IBM Storage Virtualize CLI is used to iterate over a provided set of values and changing the loop variable with each iteration. The set of values can be defined in several ways, such as a range using curly braces, a space-separated list, or command substitution. The for loop can be stopped with the Ctrl-C combination."}],
        "documents":[{"text":"Passage 1 from chapter 1.\nChapter Title: Introduction\nBook name: Do More with Less: Automating IBM Storage FlashSystem Tasks with REST APIs, Scripting, and Ansible\n# Chapter 1.  Introduction\n\nIn this chapter we discuss the following topics:\n\n- \u201cIntroduction\u201d\n- \u201cExample use cases\u201d\n- \u201cInfrastructure as Code (IaC)\u201d\n- \u201cStorage Virtualize automation opportunities\u201d\n\n\u00a9 Copyright IBM Corp. 2024.\n---\n# 1.1 Introduction\n\nAutomating the management of IBM Storage Virtualize systems (IBM Storage FlashSystem and IBM SAN Volume Controller) offers numerous benefits that can significantly enhance the efficiency and reliability of storage operations.\n\nHere is a detailed look at these benefits:\n\n# 1.1.1 Increased agility and faster storage provisioning\n\nThe following are the benefits of increased agility and faster storage provisioning.\n\n- Rapid deployment\n- Automation enables the quick deployment of storage resources, empowering organizations to adapt swiftly to evolving business needs. Instead of time-consuming manual provisioning, automation scripts and tools configure storage systems in minutes, ensuring agility and responsiveness.\n- Scalability\n- Automated storage management helps businesses scale effortlessly. As data needs evolve, storage scales automatically, preventing disruptions and ensuring consistent performance for your applications. This translates to uninterrupted business operations and a competitive edge.\n- Adaptability\n- Automation empowers IBM Storage FlashSystem to handle new workloads and applications with impressive speed. This streamlined approach ensures efficient storage support for your ever-changing business needs.\n\n# 1.1.2 Reduced human error and improved consistency\n\nBenefits of reduced human error and improved consistency are as follows.\n\n- Standardized procedures\n- Automation streamlines storage management tasks, guaranteeing consistent execution every time. Scripts and workflows eliminate the inconsistencies inherent in manual processes, leading to standardized configurations and smoother operations.\n- Error reduction\n- Manual storage management can be prone to human error, leading to downtime and data loss. Automation offers a reliable solution by minimizing errors through automating repetitive and complex tasks. This ensures consistent and reliable storage operations.\n- Compliance and best practices\n- Automation enforces adherence to industry standards and best practices, automatically ensuring all configurations meet predefined security and performance requirements. This simplifies compliance and minimizes risk for your storage environment.\n\n# 2 Do More with Less: Automating IBM Storage FlashSystem Tasks with REST APIs, Scripting, and Ansible\n---\n# 1.1.3 Simplified management of complex storage tasks\n\nThe following are the benefits of simplified management of complex storage tasks.","doc_id":0},
            {"text":"Passage 5 from chapter 1.\nChapter Title: Introduction\nBook name: Do More with Less: Automating IBM Storage FlashSystem Tasks with REST APIs, Scripting, and Ansible\nIt is available with SSH protocol and it can run bash-like scripts directly on the system. You can also include non-interactive SSH command calls into bash or PowerShell scripts running on the management host.|\n|---|---|\n|REST API|Systems have a RESTful API server that can be accessed with HTTPS protocol. It provides a set of command targets, which allows passing arguments to Storage Virtualize command line interface. Full set of CLI commands is available through the REST API, which means that absolutely every storage management task can be performed with it.|\n|Ansible playbooks|IBM provides an Ansible Collection ibm.storage_virtualize, which is a set of Ansible modules and plugins for interacting with the IBM Storage Virtualize family products to be used in your Ansible Playbooks.|\n\nAll the methods are available and consistent through all the Storage Virtualize portfolio, scripts or playbooks can be written once and used for all members of the family.\n---\n# Do More with Less: Automating IBM Storage FlashSystem Tasks with REST APIs, Scripting, and Ansible\n---","doc_id":1},
            {"text":"Passage 1 from chapter 4.\nChapter Title: Automation with REST API\nBook name: Do More with Less: Automating IBM Storage FlashSystem Tasks with REST APIs, Scripting, and Ansible\n# Chapter 4.  Automation with REST API\n\nThis chapter provides methods of managing IBM Storage Virtualize systems that can be used to integrate the system in your automation and management processes.\n\nIt describes REST API interface and provides several examples of how to interact with it.\n\nThis chapter has the following sections:\n\n- \u201cREST API on IBM Storage Virtualize\u201d on page 32\n- \u201cUsing REST API with PowerShell\u201d on page 36\n- \u201cUsing REST API in Python and Perl scripts\u201d on page 39\n\n\u00a9 Copyright IBM Corp. 2024.\n---\n# 4.1 REST API on IBM Storage Virtualize\n\nThe IBM Storage Virtualize Representational State Transfer (REST) application programming interface (API) consists of command targets that are used to retrieve system information and to create, modify, and delete system resources. These command targets allow command parameters to pass through unedited to the IBM Storage Virtualize command line interface (CLI), which handles parsing parameter specifications for validity and error reporting. Hypertext Transfer Protocol Secure (HTTPS) is used to communicate with the REST API server.\n\nAPI limits\n\nRate limiting helps with security and the prevention of an attack, such as a denial of service in which unlimited work is sent to the system. The rate limiting is implemented at second granularity and creates a return code (429 - too many requests) when a violation occurs. REST API rate limits are listed in the documentation Storage Virtualize RESTful API.\n\n# 4.1.1 REST API Explorer\n\nThe REST API Explorer is based on the Swagger UI and runs within a browser. It offers an easy way to become familiar with the API and to test the commands that it contains.\n\nThe use of the REST API Explorer requires authentication with login and password, that are used for generating a token, which is needed for all further actions. Figure 4-1 shows how to create an authentication token within the IBM SAN Volume Controller information and IBM SAN Volume Controller Task actions.\n\nTo access the REST API Explorer, enter the following URL in a browser:\n\nhttps:\/\/&lt;management_ip&gt;:7443\/rest\/explorer\n\nTo view the REST API documentation, see Storage Virtualize RESTful API. Support can also be found directly on the system within the REST API Explorer.","doc_id":2},
            {"text":"Passage 10 from chapter 5.\nChapter Title: Automation with Red Hat Ansible\nBook name: Do More with Less: Automating IBM Storage FlashSystem Tasks with REST APIs, Scripting, and Ansible\n|\n|ibm_svcinfo_command|Runs the svcinfo CLI command on the IBM Storage Virtualize system over an SSH session.|\n|ibm_svctask_command|Runs the svctask CLI commands on the IBM Storage Virtualize system over an SSH session.|\n|ibm_sv_manage_awss3_cloudaccount|Manages Amazon S3 cloud account configuration on IBM Storage Virtualize system.|\n|ibm_sv_manage_cloud_backups|Manages cloud backup on the IBM Storage Virtualize system.|\n|ibm_sv_manage_drive|This module manages drives on IBM Storage Virtualize family storage systems.|\n|ibm_sv_manage_fc_partnership|Manages Fibre Channel (FC) partnership on the IBM Storage Virtualize system.|\n|ibm_sv_manage_fcportsetmember|Manages addition or removal of ports from the Fibre Channel (FC) portsets on the IBM Storage Virtualize system.|\n|ibm_sv_manage_ip_partnership|Manages IP partnership configuration on the IBM Storage Virtualize system.|\n|ibm_sv_manage_provision_policy|Manages provisioning policies on the IBM Storage Virtualize system.|\n|ibm_sv_manage_replication_policy|Manages policy-based replication configuration on the IBM Storage Virtualize system.|\n|ibm_sv_manage_security|Manages security options on IBM Storage Virtualize family systems.|\n|ibm_sv_manage_snapshot|Manages snapshots (mutual consistent images of a volume) on the IBM Storage Virtualize system.|\n|ibm_sv_manage_snapshotpolicy|Manages snapshot policy configuration on the IBM Storage Virtualize system.|\n|ibm_sv_manage_ssl_certificate|Exports an existing system certificate on to IBM Storage Virtualize system.|\n|ibm_sv_manage_truststore_for_replication|Manages certificate trust stores for replication on the IBM Storage Virtualize system.|\n|ibm_sv_restore_cloud_backup|Restores cloud backups on IBM Storage Virtualize system.|\n|ibm_sv_switch_replication_direction|Switches the replication direction on the IBM Storage Virtualize system.|\n---\n# Note:\n\nBeginning with version 1.6.0, the ibm_svc_vdisk module is considered a deprecated feature. A new module (ibm_svc_manage_volume) was introduced to manage standard volumes.","doc_id":3},
            {"text":"Passage 24 from chapter 4.\nChapter Title: Examples and use cases\nBook name: IBM Storage Scale Information Lifecycle Management Policies\n# Example 4-40\n\nLIST policy with substitution strings POOLVAR and FILESYSVAR\n\n\/* define exclude rule*\/\nRULE 'exclude' EXCLUDE WHERE\n(PATH_NAME LIKE '%\/.SpaceMan\/%' OR\nPATH_NAME LIKE '%\/.snapshots\/%' OR\nPATH_NAME LIKE '%\/.ltfsee\/%' OR\nPATH_NAME LIKE '%\/.mmSharedTmpDir\/%' OR\nFILE_NAME like '%.mmbackupShadow%')\n\n\/* External pool rule with POOLVAR being substituted *\/\nRULE EXTERNAL POOL 'ltfs' EXEC '\/opt\/ibm\/ltfsee\/bin\/eeadm'\nOPTS 'POOLVAR' SIZE 10485760\n\n\/* Migration rule with FILESYSVAR being substituted *\/\nRULE 'MigToTape' MIGRATE FROM POOL 'silver' TO POOL 'ltfs'\nWHERE (PATH_NAME LIKE 'FILESYSVAR') AND (KB_ALLOCATED > 0)\n\nThe string 'POOLVAR' is being substituted with the name of the tape pool and the string 'FILESYSVAR' is substituted with the file system directory name subject for this policy. Note, the condition statement (PATH_NAME LIKE 'FILESYSVAR') is used to demonstrate using string substitution, it might not be required because the file system scope is given with the mmapplypolicy command in Example 4-39.\n\n# IBM Storage Scale ILM Policies\n---\n# Draft Document for Review\n\nAugust 29, 2024 10:58 am\n\n# 4.8 Quota based migration for filesets\n\nThe trigger for an automated migration is based on the occupancy of a storage pool as discussed in 2.4.1, \u201cAutomated migration\u201d. In this section we discuss an approach to trigger the migration for a fileset when a fileset quota limit (soft or hard) is met.\n\nFor this approach we use the following techniques:\n\n1. An EXTERNAL LIST policy that allows to identify files according to the quota limits using the clause THRESHOLD (resourceclass). The resource class can be FILESET_QUOTAS for hard quota limits and FILESET_QUOTA_SOFT for soft quota limits. Files are identified if the occupation in the fileset is above the quota limit. The LIST policy identifies a set of files that brings the occupation in the fileset down to a low threshold.","doc_id":4}]}
















# Prefill into base model

prompt_cache = DynamicCache()
convo = rag_chat["messages"]
docs = rag_chat["documents"]
input_text = tokenizer.apply_chat_template(conversation=convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)

inputs = tokenizer(input_text, return_tensors="pt")
with model_alora.disable_adapter():
    with torch.no_grad():
        prompt_cache = model_alora(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), past_key_values=prompt_cache).past_key_values 





# Generate aLoRA answer


input_tokenized, alora_offsets = tokenize_alora(tokenizer,input_text, INVOCATION_PROMPT)

past_key_values = copy.deepcopy(prompt_cache)
output = model_alora.generate(input_tokenized["input_ids"].to(device), attention_mask=input_tokenized["attention_mask"].to(device), use_cache=True, max_new_tokens=100, return_dict_in_generate=True, past_key_values = past_key_values, alora_offsets = alora_offsets)





output_text = tokenizer.decode(output.sequences[0])

answer = output_text.split(INVOCATION_PROMPT)[-1]
print("Answer: " + answer)
print("Golden Answer: " + convo[-1]["content"])


# from langchain_community.document_loaders import DocusaurusLoader

# # fixes a bug with asyncio and jupyter
# import nest_asyncio
# nest_asyncio.apply()

# loader = DocusaurusLoader("https://python.langchain.com")
# # Challenges
# # loader = DocusaurusLoader("https://www.tecace.com")

# docs = loader.load()

# if docs:
#     with open('output.txt', 'w', encoding='utf-8') as file:
#         for item in docs:
#             file.write(f"{item}\n")
#     print(docs[0])
# else:
#     print("No documents were loaded.")


# # with open('output.txt', 'w') as file:
# #     file.write('\n'.join(docs))

# # import csv

# # with open('output.csv', 'w', newline='') as file:
# #     writer = csv.writer(file)
# #     for item in docs:
# #         writer.writerow([item])


# # import json

# # with open('output.json', 'w') as file:
# #     json.dump(docs, file)


# docs.clear()

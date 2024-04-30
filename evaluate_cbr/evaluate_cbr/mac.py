from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc, model_pb2
from arg_services.nlp.v1 import nlp_pb2
import grpc
from glob import glob
import arguebuf as ab
from time import time

def get_text(graph: ab.Graph):
    texts = [node.label for node in graph.nodes.values() if node.label != "Support" and node.label != "Attack"]
    return " ".join(texts)

stub = retrieval_pb2_grpc.RetrievalServiceStub(grpc.insecure_channel("localhost:50200"))

def retrieve_mac(cases_path: str, query: ab.Graph, k: int = 10) -> dict[str, float]:
    files = glob(f"{cases_path}/*.json")
    cases = {f.split("/")[-1].split(".")[0]: ab.load.file(f) for f in files}
    cases = {k: model_pb2.AnnotatedGraph(graph=ab.dump.protobuf(v), text=get_text(v)) for k, v in cases.items()}
    query_ag = model_pb2.AnnotatedGraph(graph=ab.dump.protobuf(query), text=get_text(query))
    config = nlp_pb2.NlpConfig(
        language="en",
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
    )
    response = stub.Retrieve(retrieval_pb2.RetrieveRequest(semantic_retrieval=True, cases=cases, queries=[query_ag], limit=k, nlp_config=config, scheme_handling=retrieval_pb2.SchemeHandling.SCHEME_HANDLING_BINARY))
    mac_ids = {mac_graph.id: mac_graph.similarity for mac_graph in response.query_responses[0].semantic_ranking}
    return mac_ids

def retrieve_fac(cases_path, mac_results, query: ab.Graph, k: int = 10) -> dict[str, float]:
    files = glob(f"{cases_path}/*.json")
    cases = {f.split("/")[-1].split(".")[0]: ab.load.file(f) for f in files}
    cases = {k: model_pb2.AnnotatedGraph(graph=ab.dump.protobuf(v), text=get_text(v)) for k, v in cases.items()}
    # filter cases based on mac
    mac_ids = list(mac_results.keys())
    cases = {k: v for k, v in cases.items() if k in mac_ids}
    query_ag = model_pb2.AnnotatedGraph(graph=ab.dump.protobuf(query), text=get_text(query))
    config = nlp_pb2.NlpConfig(
        language="en",
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
    )
    start = time()
    response = stub.Retrieve(retrieval_pb2.RetrieveRequest(semantic_retrieval=False, structural_retrieval=True, cases=cases, queries=[query_ag], limit=k, nlp_config=config, scheme_handling=retrieval_pb2.SchemeHandling.SCHEME_HANDLING_BINARY, mapping_algorithm=retrieval_pb2.MappingAlgorithm.MAPPING_ALGORITHM_ASTAR))
    mac_ids = {mac_graph.id: mac_graph.similarity for mac_graph in response.query_responses[0].structural_ranking}
    print(f"Retrieved {len(mac_ids)} cases in {time() - start} seconds")
    return mac_ids
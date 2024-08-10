from pkg.apis.manager.v1beta1.python import api_pb2, api_pb2_grpc
from pkg.suggestion.v1beta1.internal.search_space import HyperParameter, HyperParameterSearchSpace
from pkg.suggestion.v1beta1.internal.trial import Trial, Assignment
from pkg.suggestion.v1beta1.hyperopt.base_service import BaseHyperoptService
from pkg.suggestion.v1beta1.internal.base_health_service import HealthServicer


class XGBoostService(api_pb2_grpc.SuggestionServicer, HealthServicer):
    def _generate_assignments(self, search_space: HyperParameterSearchSpace, trials: list[Trial], current_request_number: int) -> list[Assignment]:
        pass

    def ValidateAlgorithmSettings(self, request, context):
        return super().ValidateAlgorithmSettings(request, context)
    
    def GetSuggestions(self, request, context):
        search_space = HyperParameterSearchSpace.convert(request.experiment)
        trials = Trial.convert(request.trials)
        list_of_assignments = self._generate_assignments(search_space, trials, request.current_request_number)

        return api_pb2.GetSuggestionsReply(
            trials=Assignment.generate(list_of_assignments)
        )
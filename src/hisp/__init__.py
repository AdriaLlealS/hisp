from .helpers import PulsedSource, gaussian_distribution, periodic_step_function, gaussian_implantation_ufl, XDMFExportEveryDt

from .h_transport_class import CustomProblem

from .festim_models.mb_model import make_W_mb_model, make_B_mb_model, make_DFW_mb_model, make_W_mb_model_oldBC, make_B_mb_model_oldBC, make_DFW_mb_model_oldBC
from .scenario import Scenario, Pulse

from .plotting import plot_bins

from .model import Model

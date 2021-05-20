"""
metallicity prescriptions
=========================

* Strong line metallicity prescriptions from Pilyugin+2016
* Direct methods from Perez-Montero+2017, including the prescriptions
  for density and electron temperature.


"""


from .strong_line_methods import strong_line_metallicity_R, strong_line_metallicity_S
from .direct_methods import electron_density_sulfur, \
                            electron_temperature_oxygen, electron_temperature_sulfur, electron_temperature_nitrogen, \
                            oxygen_abundance_direct
from .utils import diagnostic_line_ratios

__all__ = [
            'strong_line_metallicity_R',
            'strong_line_metallicity_S',
            'electron_density_sulfur',
            'electron_temperature_oxygen',
            'electron_temperature_sulfur',
            'electron_temperature_nitrogen',
            'oxygen_abundance_direct',
            'diagnostic_line_ratios'
          ]
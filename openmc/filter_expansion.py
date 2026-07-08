class CircumferentialExpansionFilter(ExpansionFilter):
    """Abstract base class for circumferential functional expansion filters.

    This class provides common functionality for filters that expand tally
    data about the circumference of an axis-aligned cylinder. Subclasses must
    implement the order setter to define their specific bin structure.

    Parameters
    ----------
    surface : openmc.XCylinder, openmc.YCylinder, openmc.ZCylinder, or int
        The cylinder about which to take the expansion, or its ID
    order : int
        Maximum expansion order
    filter_id : int or None
        Unique identifier for the filter

    Attributes
    ----------
    surface : int
        ID of the cylinder about which the expansion is taken
    order : int
        Maximum expansion order
    id : int
        Unique identifier for the filter
    num_bins : int
        The number of filter bins

    """

    def __init__(self, surface, order, filter_id=None):
        super().__init__(order, filter_id)
        self.surface = surface

    def __hash__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tSurface', self.surface)
        return hash(string)

    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += '{: <16}=\t{}\n'.format('\tOrder', self.order)
        string += '{: <16}=\t{}\n'.format('\tSurface', self.surface)
        string += '{: <16}=\t{}\n'.format('\tID', self.id)
        return string

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, surface):
        if isinstance(surface, Integral):
            cv.check_greater_than('surface id', surface, 0, equality=True)
            self._surface = surface
        else:
            cv.check_type('surface', surface,
                          (openmc.XCylinder, openmc.YCylinder, openmc.ZCylinder))
            self._surface = surface.id

    def can_merge(self, other):
        return super().can_merge(other) and self.surface == other.surface

    def merge(self, other):
        if not self.can_merge(other):
            msg = f'Unable to merge "{type(self)}" with "{type(other)}"'
            raise ValueError(msg)
        return type(self)(self.surface, max(self.order, other.order))

    def to_xml_element(self):
        element = super().to_xml_element()
        subelement = ET.SubElement(element, 'surface')
        subelement.text = str(self.surface)
        return element

    @classmethod
    def from_xml_element(cls, elem, **kwargs):
        filter_id = int(get_text(elem, "id"))
        order = int(get_text(elem, "order"))
        surface = int(get_text(elem, "surface"))
        return cls(surface, order, filter_id=filter_id)

    @classmethod
    def from_hdf5(cls, group, **kwargs):
        if group['type'][()].decode() != cls.short_name.lower():
            raise ValueError("Expected HDF5 data for filter type '"
                             + cls.short_name.lower() + "' but got '"
                             + group['type'][()].decode() + " instead")

        filter_id = int(group.name.split('/')[-1].lstrip('filter '))
        order = group['order'][()]
        surface = int(group['surface'][()])
        return cls(surface, order, filter_id)


class CircumferentialFourierFilter(CircumferentialExpansionFilter):
    r"""Score Fourier expansion moments about a cylinder's circumference.

    This filter allows scores to be multiplied by Fourier basis functions of a
    particle's azimuthal position about an axis-aligned cylinder, up to a
    user-specified order. Only used in conjunction with a current score on the
    same surface.

    Parameters
    ----------
    surface : openmc.XCylinder, openmc.YCylinder, openmc.ZCylinder, or int
        The cylinder about which to take the expansion, or its ID
    order : int
        Maximum Fourier expansion order
    filter_id : int or None
        Unique identifier for the filter

    Attributes
    ----------
    surface : int
        ID of the cylinder about which the expansion is taken
    order : int
        Maximum Fourier expansion order
    id : int
        Unique identifier for the filter
    num_bins : int
        The number of filter bins (2*order + 1)

    """

    @ExpansionFilter.order.setter
    def order(self, order):
        ExpansionFilter.order.__set__(self, order)
        self.bins = _fourier_bin_labels(order)


class CircumferentialLegendreFilter(CircumferentialExpansionFilter):
    r"""Score Legendre expansion moments about a cylinder's circumference.

    This filter allows scores to be multiplied by Legendre polynomials of a
    particle's azimuthal position about an axis-aligned cylinder, up to a
    user-specified order. Only used in conjunction with a current score on the
    same surface.

    Parameters
    ----------
    surface : openmc.XCylinder, openmc.YCylinder, openmc.ZCylinder, or int
        The cylinder about which to take the expansion, or its ID
    order : int
        Maximum Legendre polynomial order
    filter_id : int or None
        Unique identifier for the filter

    Attributes
    ----------
    surface : int
        ID of the cylinder about which the expansion is taken
    order : int
        Maximum Legendre polynomial order
    id : int
        Unique identifier for the filter
    num_bins : int
        The number of filter bins

    """

    @ExpansionFilter.order.setter
    def order(self, order):
        ExpansionFilter.order.__set__(self, order)
        self.bins = [f'P{i}' for i in range(order + 1)]

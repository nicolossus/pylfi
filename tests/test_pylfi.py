#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pylfi
import pytest


@pytest.mark.parametrize(('n_jobs', 'exception'),
                         [('1', TypeError),
                          (1.5, TypeError),
                          (0, ValueError),
                          (-2, ValueError)])
def test_pooltools_check_and_set_jobs(n_jobs, exception):
    """Test that function raises if provided n_jobs are wrong.

    n_jobs must be given as int larger than zero (though -1 is accepted,
    this will set n_jobs automatically). The function should raise TypeError
    if wrong data type and ValueError if the value is less than or equal to
    zero (with the exception of -1).
    """
    from pylfi.utils import check_and_set_jobs

    with pytest.raises(exception):
        _ = check_and_set_jobs(n_jobs)

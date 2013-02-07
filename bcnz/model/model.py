
class mag_model:
    def find_filters(conf):

        # Demand that at least the magnitude system is specified.
        filters = []
        file_path = os.path.join(conf['data_dir'], conf['columns'])
        for line in open(file_path):
            spl = line.split()
            if not 2 < len(spl):
                continue

            filters.append(spl[0])

        filters = [re.sub('.res$','',x) for x in filters]
        filters = [x for x in filters if not x in conf['exclude']]

    #    pdb.set_trace()
        return filters

    def find_spectra(spectra_file):
        spectra = np.loadtxt(spectra_file, usecols=[0], unpack=True, dtype=np.str)
        spectra = spectra.tolist()
        spectra = [re.sub('.sed$', '', x) for x in spectra]

        return spectra

    def spectra_file(conf):
        """Detect the right path of spectra file."""

    #    for d in [conf['root'], conf['sed_dir']]:
    #    for d in [conf['root'], conf['sed_dir']]:
        file_path = os.path.join(conf['data_dir'], conf['spectra'])
        if os.path.exists(file_path):
            return file_path

        raise ValueError, 'No spectra file found.'

# -*- mode: ruby -*-
# vi: set ft=ruby :

ENV["VAGRANT_DEFAULT_PROVIDER"] = "libvirt"

Vagrant.configure("2") do |config|
  config.vm.define "kamodo-core-dev" do |config|
    config.vm.box = "potyarkin/debian11"
    config.vm.hostname = "kamodo-core-dev"

    #config.vm.network "private_network", ip: "10.3.1.20"
    #config.vm.network "public_network", :dev => "br0", :mode => "bridge", :type => "bridge"

    {
      "8050" => 8050,
      "8888" => 8888,
    }.each do |guest, host|
      config.vm.network "forwarded_port", guest: "#{guest}", host: "#{host}", host_ip: "127.0.0.1"
    end

    config.vm.synced_folder ".", "/vagrant", type: "sshfs", sshfs_opts_append: "-o cache=no", disabled: false

    config.vm.provision "Setup shell environment", type: "shell", inline: DEBIAN

    config.vm.provider :libvirt do |libvirt|
      libvirt.cpus = 4
      libvirt.memory = 4096
    end
  end
end

BEGIN {
  APT = []

  # Install docker
  APT |= ["apt-transport-https", "ca-certificates", "curl", "gnupg", "lsb-release"]
  DEBIAN_DOCKER = <<-SHELL
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
          
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
   SHELL

  # Install Miniconda
  DEBIAN_MINICONDA = <<-SHELL
    curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
    install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list
    
    apt update
    apt install -y conda
   
    source /opt/conda/etc/profile.d/conda.sh \
      && conda update conda
  SHELL

  # Install kamodo-core
  # Replace sleep with spinlock on ~/jupyter.out fd at some point.
  DEBIAN_KAMODO = <<-SHELL
    sudo -u vagrant bash -c "\
      source /opt/conda/etc/profile.d/conda.sh \
        && conda init \
        && conda create -n kamodo python==3.7 \
        && conda activate kamodo \
        && conda install jupyter \
        && pip install kamodo-core mkdocs python-markdown-math markdown-include mknotebooks \
        && (jupyter notebook --no-browser --ip 0.0.0.0 /vagrant/docs/notebooks > ~/jupyter.out 2>&1 &) \
        && sleep 10 \
        && jupyter notebook list"
   
    sudo -u vagrant bash -c "echo 'conda activate kamodo' >> ~/.bashrc"
  SHELL

  # Install Debian
  DEBIAN = <<-SHELL
    export DEBIAN_FRONTEND=noninteractive
 
    apt-get update
    apt-get upgrade
    apt-get install -y #{APT.join(" ")}
 
    #{DEBIAN_DOCKER}
    #{DEBIAN_MINICONDA}

    #{DEBIAN_KAMODO}
  SHELL
}
